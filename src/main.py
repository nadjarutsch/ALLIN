import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')
    
import matplotlib.pyplot as plt
from collections import defaultdict

import networkx as nx
import causaldag 
import cdt

import data_generation.causal_graphs.graph_generation as graph_gen
import data_generation.causal_graphs.graph_visualization as visual
import data_generation.data_generation as data_gen
import data_generation.datasets as data
import models.multivar_mlp as mmlp
import causal_discovery.prepare_data as cd
import outlier_detection.model_based as ood
import outlier_detection.depcon_kernel as depcon
import metrics
import clustering.dbscan as dbscan
import clustering.kmeans as kmeans
from fci import FCI
import sklearn
from sklearn.cluster import DBSCAN
from itertools import product
from itertools import combinations
import random
import yaml
# from clustering.dbscan_hparam import get_eps
import hdbscan



N_OBS = 1000 # overwritten through hydra
INT_RATIO = 1
BATCH_SIZE = 128
lr = 1e-3
loss = mmlp.nll
epochs = 5
fit_epochs = 60
stds = 3
seeds = list(range(50))
# seeds = [random.randint(0, 100)]
NUM_VARS = 5
true_target_indices = np.cumsum([N_OBS] + [INT_RATIO * N_OBS] * NUM_VARS)
alpha_skeleton = 0.01
alpha = 0.00001
expected_N = 2

# os.environ['WANDB_MODE'] = 'offline'


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # wandb config for logging
    config = dict(
        n_obs = cfg.n_obs,
        int_ratio = INT_RATIO,
        batch_size = BATCH_SIZE,
        lr = lr,
        epochs = epochs,
        fit_epochs = fit_epochs,
        threshold = stds,
        num_vars = NUM_VARS,
        graph_structure = 'random',
        edge_prob = cfg.expected_N / NUM_VARS,
        E_N = cfg.expected_N,
        mu = 0.0,
        sigma = cfg.int_sigma,
        minpts = cfg.minpts,
        eps = cfg.eps,
        citest = 'gaussian',
        alpha_skeleton = alpha_skeleton,
        alpha = alpha,
        num_clus = NUM_VARS + 1,
        int_mu = cfg.int_mu,
        int_sigma = cfg.int_sigma,
        clustering = cfg.clustering,
        n_int_targets = NUM_VARS,
        cluster_metric = cfg.cluster_metric
    )
    
    
    if torch.cuda.is_available():
        cdt.SETTINGS.rpath = '/sw/arch/Debian10/EB_production/2021/software/R/4.1.0-foss-2021a/lib/R/bin/Rscript'
        device = 'cuda:0'
    else:
        cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'
        device = 'cpu'

    
    for seed in seeds:
        config['seed'] = seed
        run = wandb.init(project="idiod", entity="nadjarutsch", group=cfg.group, notes='', tags=[], config=config, reinit=True)
        with run:

            ### DATA GENERATION ###

            # generate graph
            dag = data_gen.generate_dag(num_vars=config['num_vars'], edge_prob=config['edge_prob'], fns='linear gaussian', mu=config['mu'], sigma=config['sigma'])
            variables = [v.name for v in dag.variables]
            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)

            # plot the true underlying causal graph
            true_graph = dag.nx_graph
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(true_graph)
            nx.draw(true_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"true graph": wandb.Image(plt)})
            plt.close()

            # get true essential graph representing the MEC
            adj_matrix, var_lst = causaldag.DAG.from_nx(true_graph).cpdag().to_amat()
            mapping = dict(zip(range(len(var_lst)), var_lst))
            mec = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            mec = nx.relabel_nodes(mec, mapping)
            visual.set_edge_attributes(mec)
            # logging
            plt.figure(figsize=(6, 6))
            colors = visual.get_colors(mec)
            nx.draw(mec, with_labels=True, node_size=1000, node_color='w', edgecolors='black', edge_color=colors)
            wandb.log({"essential graph": wandb.Image(plt)})
            plt.close()

            # generate data
            synth_dataset, interventions = data_gen.generate_data(dag=dag, n_obs=config['n_obs'], int_ratio=INT_RATIO, seed=seed, int_mu=config['int_mu'], int_sigma=config['int_sigma'])

            # get_eps(10, synth_dataset.features)
            # get_eps(50, synth_dataset.features)
            # get_eps(500, synth_dataset.features)

            '''### ORACLE PC+CONTEXT ###

            # create the PC+context graph
            pc_context_graph = true_graph.copy()

            context_vars = []
            for var in random.sample(variables, config["n_int_targets"]):
                pc_context_graph.add_node(f'I_{var}')
                pc_context_graph.add_edge(f'I_{var}', var)
                context_vars.append(f'I_{var}')

            pc_context_graph.add_node('conf')
            for var in context_vars:
                pc_context_graph.add_edge('conf', var)

            plt.figure(figsize=(6, 6))
            colors = visual.get_colors(pc_context_graph)
            nx.draw(pc_context_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black', edge_color=colors)
            wandb.log({"PC+context graph": wandb.Image(plt)})
            plt.close()

            # get essential graph of the PC+context graph
            adj_matrix, var_lst = causaldag.DAG.from_nx(pc_context_graph).cpdag().to_amat()
            mapping = dict(zip(range(len(var_lst)), var_lst))
            pc_context_mec = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            pc_context_mec = nx.relabel_nodes(pc_context_mec, mapping)
            visual.set_edge_attributes(pc_context_mec)
            # logging
            plt.figure(figsize=(6, 6))
            colors = visual.get_colors(pc_context_mec)
            nx.draw(pc_context_mec, with_labels=True, node_size=1000, node_color='w', edgecolors='black', edge_color=colors)
            wandb.log({"Oracle PC+context MEC": wandb.Image(plt)})
            plt.close()

            # remove context variables and compute metrics
            pc_context_mec.remove_nodes_from(context_vars + ['conf'])
            wandb.run.summary["Oracle PC+context: SHD"] = cdt.metrics.SHD(true_graph, pc_context_mec,
                                                                   double_for_anticausal=False)
            wandb.run.summary["Oracle PC+context: SID"] = cdt.metrics.SID(true_graph, pc_context_mec)
            wandb.run.summary["Oracle PC+context: CC"] = metrics.causal_correctness(true_graph, pc_context_mec, mec)

            '''### FEATURE TRANSFORMATION ###

            '''gnmodel = mmlp.GaussianNoiseModel(num_vars=config["num_vars"], hidden_dims=[])
            optimizer = torch.optim.Adam(gnmodel.parameters(), lr=config["lr"])
            loss = mmlp.nll
            fit_adj_matrix = torch.ones((config["num_vars"], config["num_vars"]))
            fit_adj_matrix.fill_diagonal_(0)

            partitions = ood.cluster(synth_dataset,
                                     gnmodel,
                                     loss,
                                     optimizer,
                                     config["epochs"],
                                     config["fit_epochs"],
                                     fit_adj_matrix,
                                     config["threshold"])
            #ood.fit(synth_dataset.partitions[0], gnmodel, loss, optimizer, config["fit_epochs"], config["batch_size"], fit_adj_matrix)
            for part in partitions:
                print(len(part))

            preds = gnmodel(synth_dataset.features[..., :-1], fit_adj_matrix)
            losses = loss(preds, synth_dataset.features[..., :-1]).detach()

            true_adj_matrix = nx.to_numpy_array(true_graph)
            root_vars = torch.nonzero(torch.all(~torch.from_numpy(true_adj_matrix).bool(), dim=0))
            cond_targets = [0 if label - 1 in root_vars else label for label in synth_dataset.targets]
            clustering_dataset = data.PartitionData(features=losses, targets=cond_targets)
            '''
            clustering_dataset = synth_dataset

            ### CAUSAL DISCOVERY BEFORE CLUSTERING ###

            # correct partitions
            target_dataset = data.PartitionData(features=synth_dataset.features[..., :-1],
                                                targets=synth_dataset.targets)
            target_dataset.update_partitions(target_dataset.targets)
            # obs_dataset = data.PartitionData(features=target_dataset.partitions[0].features[...,:-1])

            # what happens if all data comes from the same (observational) distribution?
            # synth_dataset = obs_dataset
            # clustering_dataset = obs_dataset # TODO: redundant, make nice

            '''# PC on ground truth clusters
            fps = []
            fns = []
            shds = []
            for i, cluster in zip(set(target_dataset.targets), target_dataset.partitions):
                cluster_dataset = data.PartitionData(features=cluster.features[..., :-1])
                df = cd.prepare_data(cd="pc", data=cluster_dataset, variables=variables)

                model_pc = cdt.causality.graph.PC(CItest="gaussian", alpha=config["alpha"])
                created_graph = model_pc.predict(df)

                plt.figure(figsize=(6, 6))
                colors = visual.get_colors(created_graph)
                nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black',
                        edge_color=colors)
                wandb.log({f"ground truth cluster {i}": wandb.Image(plt)})
                plt.close()

                # true graph of the matched interventional distribution
                int_adj_matrix = nx.to_numpy_array(true_graph)
                if i > 0:
                    int_adj_matrix[:, i - 1] = 0
                true_int_graph = nx.from_numpy_array(int_adj_matrix, create_using=nx.DiGraph)
                true_int_graph = nx.relabel_nodes(true_int_graph, mapping)

                fps.append(metrics.fp(created_graph, mec))
                fns.append(metrics.fn(created_graph, mec))
                shds.append(cdt.metrics.SHD(true_int_graph, created_graph, double_for_anticausal=False))

            wandb.run.summary["Target clusters: avg FP"] = np.mean(fps)
            wandb.run.summary["Target clusters: avg FN"] = np.mean(fns)
            wandb.run.summary["Target clusters: SHD"] = np.mean(shds)'''


            # initial causal discovery (skeleton)
            df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            # pc algorithm test on observational data only
            # df = cd.prepare_data(cd="pc", data=obs_dataset, variables=variables)

            model_pc = cdt.causality.graph.PC(alpha=config["alpha"], CItest=config["citest"])
            # model_fci = FCI(alpha=config["alpha_skeleton"], CItest=config["citest"])
            # skeleton = model_fci.create_graph_from_data(df)
            skeleton = model_pc.predict(df)
            
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(skeleton)
            nx.draw(skeleton, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"skeleton": wandb.Image(plt)})
            plt.close()

            wandb.run.summary["PC: SHD to MEC"] = cdt.metrics.SHD(mec, skeleton, double_for_anticausal=False)
            wandb.run.summary["PC: SHD"] = cdt.metrics.SHD(true_graph, skeleton, double_for_anticausal=False)
            wandb.run.summary["PC: SID"] = cdt.metrics.SID(true_graph, skeleton)
            wandb.run.summary["PC: CC"] = metrics.causal_correctness(true_graph, skeleton, mec)
            
            
            # use inferred skeleton
            # adj_matrix = torch.from_numpy(nx.to_numpy_array(skeleton))
            
            # use true skeleton
            # adj_matrix = torch.from_numpy(nx.to_numpy_array(mec))


            # intervention detection (ood)
        #    print('Creating model...')
        #    gnmodel = mmlp.GaussianNoiseModel(num_vars=dag.num_vars, hidden_dims=[])
        #    optimizer = torch.optim.Adam(gnmodel.parameters(), lr=lr)
        #    partitions_obs = ood.cluster(synth_dataset, gnmodel, loss, optimizer, epochs, fit_epochs, adj_matrix, stds, BATCH_SIZE)


            ### CLUSTERING ###

            if config["clustering"] == "depcon kmeans":
                # kernel K-means
                labels = depcon.kernel_k_means(clustering_dataset.features[...,:-1], init='k-means++', num_clus=config['num_clus'], device=device)

            elif config["clustering"] == "kmeans":
                # normal K-means
                labels = kmeans.kmeans(clustering_dataset.features[...,:-1], init='k-means++', n_clusters=len(set(synth_dataset.targets))) # TODO: number of clusters automatically

            elif config["clustering"] == "dbscan":
                # DBSCAN clustering
              #  kappa, gamma = depcon.dep_contrib_kernel(synth_dataset.features[...,:-1], device=device)
              #  distance_matrix = torch.arccos(kappa).cpu().detach()
              #  partitions = dbscan.dbscan(distance_matrix, minpts=config["minpts"], metric="precomputed")
              #  synth_dataset.update_partitions(partitions)
                labels = DBSCAN(eps=config["eps"], min_samples=config["minpts"]).fit(clustering_dataset.features[...,:-1]).labels_

            elif config["clustering"] == "hdbscan":
                # HDBSCAN*
                labels = hdbscan.HDBSCAN(min_cluster_size=config["minpts"], metric=config["cluster_metric"]).fit(clustering_dataset.features[...,:-1]).labels_

            synth_dataset.update_partitions(labels)
            wandb.log({"cluster sizes": wandb.Histogram(labels)})

            # cluster analysis
            # (1) avg sample likelihood
            # metrics.joint_log_prob(dataset=synth_dataset, dag=dag, interventions=interventions, title="K-means clusters")

            # likelihood evaluation for ground truth partitions (optimal)
            # metrics.joint_log_prob(dataset=target_dataset, dag=dag, interventions=interventions, title="Ground truth distributions")

            # (2) ARI, AMI, NMI (standard cluster evaluation metrics)
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(synth_dataset.targets, labels)
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(synth_dataset.targets, labels)
            wandb.run.summary["NMI"] = sklearn.metrics.normalized_mutual_info_score(synth_dataset.targets, labels)

            '''
            # Match clusters to intervention targets

            counts = []
            int_targets = []
            for cluster, target in product(synth_dataset.partitions, target_dataset.partitions):
                # compare equal elements
                count = len(set(cluster.features[..., -1].tolist()) & set(target.features[..., -1].tolist()))
                counts.append(count)
                if len(counts) == len(target_dataset.partitions):
                    int_targets.append(list(set(synth_dataset.targets))[np.argmax(counts)])
                    counts = []
            '''

            ### CAUSAL DISCOVERY ###
            '''
            # PC on each partition separately

            shds = []
            fps = []
            fns = []
            for i, cluster in enumerate(synth_dataset.partitions):
                cluster_dataset = data.PartitionData(features=cluster.features[..., :-1])
                df = cd.prepare_data(cd="pc", data=cluster_dataset, variables=variables)
                model_pc = cdt.causality.graph.PC(CItest=config["citest"], alpha=config["alpha"])
                created_graph = model_pc.predict(df)

                plt.figure(figsize=(6, 6))
                colors = visual.get_colors(created_graph)
                nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black', edge_color=colors)
                wandb.log({f"predicted graph, cluster {i}": wandb.Image(plt)})
                plt.close()

                # true graph of the matched interventional distribution
                int_adj_matrix = nx.to_numpy_array(true_graph)
                if i > 0:
                    int_adj_matrix[:,int_targets[i]-1] = 0
                true_int_graph = nx.from_numpy_array(int_adj_matrix, create_using=nx.DiGraph)
                true_int_graph = nx.relabel_nodes(true_int_graph, mapping)

                fps.append(metrics.fp(created_graph, mec))
                fns.append(metrics.fn(created_graph, mec))
                shds.append(cdt.metrics.SHD(true_int_graph, created_graph, double_for_anticausal=False))

            wandb.run.summary["Pred clusters: avg FP"] = np.mean(fps)
            wandb.run.summary["Pred clusters: avg FN"] = np.mean(fns)
            wandb.run.summary["Pred clusters: SHD"] = np.mean(shds)
            '''

            # putting everything together: PC with context variables
            synth_dataset.set_random_intervention_targets()
            df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)

            # logging
            # tbl = wandb.Table(dataframe=df)
            # wandb.log({"clustered data": tbl})
    
            # for node in list(df.columns.values[config['num_vars']:]):
            #    skeleton.add_node(node)
            #    skeleton.add_edge(node, node.replace("I_",""))
    
            model_pc = cdt.causality.graph.PC(CItest=config["citest"], alpha=config["alpha"])
            created_graph = model_pc.predict(df)
            created_graph.remove_nodes_from(list(df.columns.values[config['num_vars']:])) # TODO: doublecheck

            wandb.run.summary["PC+context: SHD to MEC"] = cdt.metrics.SHD(mec, created_graph, double_for_anticausal=False)
            wandb.run.summary["PC+context: SHD"] = cdt.metrics.SHD(true_graph, created_graph, double_for_anticausal=False)
            wandb.run.summary["PC+context: SID"] = cdt.metrics.SID(true_graph, created_graph)
            wandb.run.summary["PC+context: CC"] = metrics.causal_correctness(true_graph, created_graph, mec)
    
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(created_graph)
            nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"PC+context, pred clusters": wandb.Image(plt)})
            plt.close()

            # target partitions
            target_dataset.set_random_intervention_targets()
            df_target = cd.prepare_data(cd="pc", data=target_dataset, variables=variables)

            tbl = wandb.Table(dataframe=df_target)
            wandb.log({"clustered data (target)": tbl})

            model_pc = cdt.causality.graph.PC(CItest=config["citest"], alpha=config["alpha"])
            created_graph = model_pc.predict(df_target)
            created_graph.remove_nodes_from(list(df_target.columns.values[config['num_vars']:]))  # TODO: doublecheck

            wandb.run.summary["PC+context target: SHD"] = cdt.metrics.SHD(true_graph, created_graph, double_for_anticausal=False)
            wandb.run.summary["PC+context target: SID"] = cdt.metrics.SID(true_graph, created_graph)
            wandb.run.summary["PC+context target: CC"] = metrics.causal_correctness(true_graph, created_graph, mec)

            plt.figure(figsize=(6, 6))
            colors = visual.get_colors(created_graph)
            nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black',
                    edge_color=colors)
            wandb.log({"PC+context, target clusters": wandb.Image(plt)})
            plt.close()

            '''
            # JCI
            model_jci = FCI(alpha=config["alpha"], CItest=config["citest"])
            contextvars = range(len(variables), len(variables) + len(synth_dataset.partitions))
            jci_graph = model_jci.predict(df, jci="123", contextvars=contextvars)
            jci_graph.remove_nodes_from(list(df.columns.values[config['num_vars']:]))  # TODO: doublecheck

            wandb.run.summary["JCI pred: SHD"] = cdt.metrics.SHD(true_graph, jci_graph, double_for_anticausal=False)
            wandb.run.summary["JCI pred: SID"] = cdt.metrics.SID(true_graph, jci_graph)
            wandb.run.summary["JCI pred: CC"] = metrics.causal_correctness(true_graph, jci_graph, mec)

            plt.figure(figsize=(6, 6))
            colors = visual.get_colors(jci_graph)
            nx.draw(jci_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black',
                    edge_color=colors)
            wandb.log({"JCI, pred clusters": wandb.Image(plt)})
            plt.close()


            model_jci = FCI(alpha=config["alpha"], CItest=config["citest"])
            jci_target_graph = model_jci.predict(df_target, jci="123", contextvars=list(
                range(len(variables), len(variables) + len(target_dataset.partitions))))
            jci_target_graph.remove_nodes_from(list(df_target.columns.values[config['num_vars']:]))  # TODO: doublecheck

            wandb.run.summary["JCI target: SHD"] = cdt.metrics.SHD(true_graph, jci_target_graph, double_for_anticausal=False)
            wandb.run.summary["JCI target: SID"] = cdt.metrics.SID(true_graph, jci_target_graph)
            wandb.run.summary["JCI target: CC"] = metrics.causal_correctness(true_graph, jci_target_graph, mec)

            plt.figure(figsize=(6, 6))
            colors = visual.get_colors(jci_target_graph)
            nx.draw(jci_target_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black',
                    edge_color=colors)
            wandb.log({"JCI, target clusters": wandb.Image(plt)})
            plt.close()

            '''
            wandb.finish()



if __name__ == '__main__':
    main()
    # sweep_config = yaml.safe_load(open('sweep.yaml', 'r'))
    # sweep_id = wandb.sweep(sweep_config, project="idiod")
    # wandb.agent(sweep_id, main, count=800)