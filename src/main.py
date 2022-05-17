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


N_OBS = 10000
INT_RATIO = 0.01
BATCH_SIZE = 128
lr = 1e-3
loss = mmlp.nll
epochs = 5
fit_epochs = 60
stds = 4
seeds = list(range(50))
NUM_VARS = 7
true_target_indices = np.cumsum([N_OBS] + [INT_RATIO * N_OBS] * NUM_VARS)
alpha_skeleton = 0.01
alpha = 0.00001
expected_N = 2


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
        E[N] = cfg.expected_N,
        mu = 0.0,
        sigma = 1.,
        minpts = 5,
        citest = 'gaussian',
        alpha_skeleton = alpha_skeleton,
        alpha = alpha
    )
    
    
    if torch.cuda.is_available():
        cdt.SETTINGS.rpath = '/sw/arch/Debian10/EB_production/2021/software/R/4.1.0-foss-2021a/lib/R/bin/Rscript'
        device = 'cuda:0'
    else:
        cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'
        device = 'cpu'

    
    for seed in seeds:
        config['seed'] = seed
        run = wandb.init(project="idiod", entity="nadjarutsch", group='pc reproduction', notes='normal distributions', tags=['pc', 'kalisch2007'], config=config, reinit=True)
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

            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(true_graph)
            
            synth_dataset, interventions = data_gen.generate_data(dag=dag, n_obs=N_OBS, int_ratio=INT_RATIO, seed=seed)

            # correct partitions
            target_dataset = data.PartitionData(features=synth_dataset.features[..., 0], targets=synth_dataset.targets)
            target_dataset.update_partitions(target_dataset.targets)
            obs_dataset = data.PartitionData(features=target_dataset.partitions[0].features[..., 0], targets=target_dataset.targets)

            # initial causal discovery (skeleton)
          #  df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            # pc algorithm test on observational data only
            df = cd.prepare_data(cd="pc", data=obs_dataset, variables=variables)

            model_pc = cdt.causality.graph.PC(alpha=config["alpha_skeleton"], CItest=config["citest"])
            skeleton = model_pc.create_graph_from_data(df) 
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
            adj_matrix = torch.from_numpy(nx.to_numpy_array(mec))
    
            # intervention detection (ood)
        #    print('Creating model...')
        #    gnmodel = mmlp.GaussianNoiseModel(num_vars=dag.num_vars, hidden_dims=[])
        #    optimizer = torch.optim.Adam(gnmodel.parameters(), lr=lr)
        #    partitions_obs = ood.cluster(synth_dataset, gnmodel, loss, optimizer, epochs, fit_epochs, adj_matrix, stds, BATCH_SIZE)
            
            '''# set ground truth observational and interventional partitions
            synth_dataset.update_partitions([list(range(N_OBS)), list(range(N_OBS, int(N_OBS + config['num_vars'] * N_OBS * INT_RATIO)))])
                
            # evaluate causal kernel on the interventional partition
            partitions_temp = depcon.kernel_k_means(synth_dataset.partitions[1].features[...,0], num_clus=config['num_vars'], device=device)
            partitions_int = []
            for part in partitions_temp:
                partitions_int.append([p + N_OBS for p in part if p])

            partitions = [list(range(N_OBS))]
            partitions.extend(partitions_int)
            synth_dataset.update_partitions(partitions)
            
            # kernel analysis         
            # (1) avg sample likelihood
                
            metrics.joint_log_prob(dataset=synth_dataset, dag=dag, interventions=interventions, title="K-means clusters")
            '''
            # DBSCAN clustering
          #  kappa, gamma = depcon.dep_contrib_kernel(synth_dataset.features[...,0], device=device)
          #  distance_matrix = torch.arccos(kappa).cpu().detach()
          #  partitions = dbscan.dbscan(distance_matrix, minpts=config["minpts"], metric="precomputed")
          #  synth_dataset.update_partitions(partitions)
          #  metrics.joint_log_prob(dataset=synth_dataset, dag=dag, interventions=interventions, title="DBSCAN clusters")

            # likelihood evaluation for ground truth partitions (optimal)
            # metrics.joint_log_prob(dataset=target_dataset, dag=dag, interventions=interventions, title="Ground truth distributions")

            
            '''borders = true_target_indices.tolist()
            borders.insert(0,0)

            for p_i, part in enumerate(partitions_int):
                counts = []
                for i in range(len(partitions)):
                    idx_lower = borders[i]
                    idx_upper = borders[i+1]

                    print([(p >= idx_lower) and (p < idx_upper) for p in part])
                    counts.append(sum([(p >= idx_lower) and (p < idx_upper) for p in part]))
                                    

                fig = plt.figure(figsize=(6,6))
                ax = fig.add_axes([0,0,1,1])
                partitions = list(range(len(partitions)))
                ax.bar(partitions, counts)
                wandb.log({f"partition_{p_i}": wandb.Image(plt)})
                plt.close()'''
            
            
            # causal discovery
            '''synth_dataset.set_true_intervention_targets(true_target_indices)
            
            df = cd.prepare_data(cd="pc", data=synth_dataset, variables=variables)
            
            # logging
            tbl = wandb.Table(dataframe=df)
            wandb.log({"clustered data": tbl})
    
            for node in list(df.columns.values[config['num_vars']:]):
                skeleton.add_node(node)
                skeleton.add_edge(node, node.replace("I_",""))
    
            model_pc = cdt.causality.graph.PC(CItest="rcot", alpha=config["alpha"])
            created_graph = model_pc.orient_directed_graph(df, skeleton)
            created_graph.remove_nodes_from(list(df.columns.values[config['num_vars']:]))
            
            wandb.run.summary["SHD"] = cdt.metrics.SHD(true_graph, created_graph, double_for_anticausal=False)  
            wandb.run.summary["SID"] = cdt.metrics.SID(true_graph, created_graph)
            wandb.run.summary["CC"] = metrics.causal_correctness(true_graph, created_graph, mec)
    
            plt.figure(figsize=(6,6))
            colors = visual.get_colors(created_graph)
            nx.draw(created_graph, with_labels=True, node_size=1000, node_color='w', edgecolors ='black', edge_color=colors)
            wandb.log({"discovered graph": wandb.Image(plt)})
            plt.close()'''
    
    




if __name__ == '__main__':
    
    main()