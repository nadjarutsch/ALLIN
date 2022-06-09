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


N_OBS = 1000 # overwritten through hydra
INT_RATIO = 1
BATCH_SIZE = 128
lr = 1e-3
loss = mmlp.nll
epochs = 5
fit_epochs = 60
stds = 4
seeds = list(range(10))
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
        threshold = f'{stds} stds',
        num_vars = NUM_VARS,
        graph_structure = 'random',
        edge_prob = cfg.expected_N / NUM_VARS,
        E_N = cfg.expected_N,
        mu = 0.0,
        sigma = cfg.int_sigma,
        minpts = 5,
        citest = 'gaussian',
        alpha_skeleton = alpha_skeleton,
        alpha = alpha,
        num_clus = NUM_VARS + 1,
        int_mu = cfg.int_mu,
        int_sigma = cfg.int_sigma
    )
    
    
    if torch.cuda.is_available():
        cdt.SETTINGS.rpath = '/sw/arch/Debian10/EB_production/2021/software/R/4.1.0-foss-2021a/lib/R/bin/Rscript'
        device = 'cuda:0'
    else:
        cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'
        device = 'cpu'

    
    for seed in seeds:
        config['seed'] = seed
        run = wandb.init(project="idiod", entity="nadjarutsch", group='kmeans++, varying mean and both std', notes='', tags=['kmeans', 'kmeans++', 'depcon'], config=config, reinit=True)
        with run:
            # generate data
            dag = data_gen.generate_dag(num_vars=config['num_vars'], edge_prob=config['edge_prob'], fns='linear gaussian', mu=config['mu'], sigma=config['sigma'])
            variables = [v.name for v in dag.variables]

            true_graph = dag.nx_graph
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(true_graph)
            nx.draw(true_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"true graph": wandb.Image(plt)})
            plt.close()

            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)
            
            synth_dataset, interventions = data_gen.generate_data(dag=dag, n_obs=config['n_obs'], int_ratio=INT_RATIO, seed=seed, int_mu=config['int_mu'], int_sigma=config['int_sigma'])

            # correct partitions
            target_dataset = data.PartitionData(features=synth_dataset.features[...,:-1], targets=synth_dataset.targets)
            target_dataset.update_partitions(target_dataset.targets)
            # obs_dataset = data.PartitionData(features=target_dataset.partitions[0].features[...,:-1])

            # PC on ground truth clusters
            for i, cluster in enumerate(target_dataset.partitions):
                cluster_dataset = data.PartitionData(features=cluster.features[...,:-1])
                df = cd.prepare_data(cd="pc", data=cluster_dataset, variables=variables)
                model_pc = cdt.causality.graph.PC(CItest="gaussian", alpha=config["alpha"])
                created_graph = model_pc.predict(df)

                plt.figure(figsize=(6, 6))
                colors = visual.get_colors(created_graph)
                nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black',
                        edge_color=colors)
                wandb.log({f"ground truth cluster {i}": wandb.Image(plt)})
                plt.close()

            '''
            # initial causal discovery (skeleton)
            df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            # pc algorithm test on observational data only
            df = cd.prepare_data(cd="pc", data=obs_dataset, variables=variables)

           # model_pc = cdt.causality.graph.PC(alpha=config["alpha_skeleton"], CItest=config["citest"])
            model_fci = FCI(alpha=config["alpha_skeleton"], CItest=config["citest"])
            skeleton = model_fci.create_graph_from_data(df)

            adj_matrix, var_lst = causaldag.DAG.from_nx(true_graph).cpdag().to_amat()
            mapping = dict(zip(range(len(var_lst)), var_lst))
            mec = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            mec = nx.relabel_nodes(mec, mapping)
            visual.set_edge_attributes(mec)
            
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(skeleton)
            nx.draw(skeleton, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"skeleton": wandb.Image(plt)})
            plt.close()
            
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(mec)
            nx.draw(mec, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"true skeleton": wandb.Image(plt)})
            plt.close()

            wandb.run.summary["skeleton SHD"] = cdt.metrics.SHD(mec, skeleton, double_for_anticausal=False)

            # use inferred skeleton
            # adj_matrix = torch.from_numpy(nx.to_numpy_array(skeleton))
            
            # use true skeleton
            adj_matrix = torch.from_numpy(nx.to_numpy_array(mec))'''
    
            # intervention detection (ood)
        #    print('Creating model...')
        #    gnmodel = mmlp.GaussianNoiseModel(num_vars=dag.num_vars, hidden_dims=[])
        #    optimizer = torch.optim.Adam(gnmodel.parameters(), lr=lr)
        #    partitions_obs = ood.cluster(synth_dataset, gnmodel, loss, optimizer, epochs, fit_epochs, adj_matrix, stds, BATCH_SIZE)
            
            '''# set ground truth observational and interventional partitions
            synth_dataset.update_partitions([list(range(cfg.n_obs)), list(range(cfg.n_obs, int(cfg.n_obs + config['num_vars'] * cfg.n_obs * INT_RATIO)))])
            '''

            # kernel K-means
            # labels = depcon.kernel_k_means(synth_dataset.features[...,:-1], init='k-means++', num_clus=config['num_clus'], device=device)
            # synth_dataset.update_partitions(labels)

            # normal K-means
            labels = kmeans.kmeans(synth_dataset.features[...,:-1], init='k-means++', n_clusters=config['num_clus'])
            synth_dataset.update_partitions(labels)

            # DBSCAN clustering
          #  kappa, gamma = depcon.dep_contrib_kernel(synth_dataset.features[...,:-1], device=device)
          #  distance_matrix = torch.arccos(kappa).cpu().detach()
          #  partitions = dbscan.dbscan(distance_matrix, minpts=config["minpts"], metric="precomputed")
          #  synth_dataset.update_partitions(partitions)
          #  metrics.joint_log_prob(dataset=synth_dataset, dag=dag, interventions=interventions, title="DBSCAN clusters")

            # cluster analysis
            # (1) avg sample likelihood
            # metrics.joint_log_prob(dataset=synth_dataset, dag=dag, interventions=interventions, title="K-means clusters")

            # likelihood evaluation for ground truth partitions (optimal)
            # metrics.joint_log_prob(dataset=target_dataset, dag=dag, interventions=interventions, title="Ground truth distributions")

            # (2) ARI, AMI, NMI (standard cluster evaluation metrics)
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(synth_dataset.targets, labels)
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(synth_dataset.targets, labels)
            wandb.run.summary["NMI"] = sklearn.metrics.normalized_mutual_info_score(synth_dataset.targets, labels)


            # causal discovery
            # synth_dataset.set_true_intervention_targets(true_target_indices)

            # PC on each partition separately
            '''
            for i, cluster in enumerate(synth_dataset.partitions):
                cluster_dataset = data.PartitionData(features=cluster.features[..., :-1])
                df = cd.prepare_data(cd="pc", data=cluster_dataset, variables=variables)
                model_pc = cdt.causality.graph.PC(CItest="rcot", alpha=config["alpha"])
                created_graph = model_pc.predict(df)

                plt.figure(figsize=(6, 6))
                colors = visual.get_colors(created_graph)
                nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors='black', edge_color=colors)
                wandb.log({f"predicted graph, cluster {i}": wandb.Image(plt)})
                plt.close()'''


            # df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            
            # logging
            # tbl = wandb.Table(dataframe=df)
            # wandb.log({"clustered data": tbl})
    
            # for node in list(df.columns.values[config['num_vars']:]):
            #    skeleton.add_node(node)
            #    skeleton.add_edge(node, node.replace("I_",""))
    
            # model_pc = cdt.causality.graph.PC(CItest="rcot", alpha=config["alpha"])
            # created_graph = model_pc.orient_directed_graph(df, skeleton)
            # created_graph.remove_nodes_from(list(df.columns.values[config['num_vars']:]))
            
            # wandb.run.summary["SHD"] = cdt.metrics.SHD(true_graph, created_graph, double_for_anticausal=False)
            # wandb.run.summary["SID"] = cdt.metrics.SID(true_graph, created_graph)
            # wandb.run.summary["CC"] = metrics.causal_correctness(true_graph, created_graph, mec)
    
            # plt.figure(figsize=(6,6))
            # colors = visual.get_colors(created_graph)
            # nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            # wandb.log({"discovered graph": wandb.Image(plt)})
            # plt.close()

            wandb.finish()



if __name__ == '__main__':
    main()