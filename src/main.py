import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import copy

import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')

from collections import defaultdict

import networkx as nx
# import causaldag
import cdt

import data_generation.causal_graphs.graph_generation as graph_gen

import data_generation.data_generation as data_gen
import data_generation.datasets as data
import causal_discovery.prepare_data as cd
import outlier_detection.depcon_kernel as depcon
import metrics
from fci import FCI
from plotting import plot_graph
from data_generation.causal_graphs.graph_utils import dag_to_mec, add_context_vars, get_interventional_graph

import sklearn
from sklearn.cluster import KMeans
from clustering.utils import *
import hdbscan



os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['WANDB_MODE'] = 'offline'

@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    
    if torch.cuda.is_available():
        cdt.SETTINGS.rpath = '/sw/arch/Debian10/EB_production/2021/software/R/4.1.0-foss-2021a/lib/R/bin/Rscript'
        cfg.device = 'cuda:0'
    else:
        cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'
        cfg.device = 'cpu'

    
    for seed in range(cfg.start_seed, cfg.end_seed):
        cfg.seed = seed
        if str(cfg.clustering.name) == "kmeans":  # TODO: with resolver (hydra)
            cfg.clustering.clusterer.n_clusters = cfg.graph.num_vars + 1
        if "gmm" in str(cfg.clustering.name):
            cfg.clustering.clusterer.n_components = cfg.graph.num_vars + 1
        run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, group=cfg.wandb.group, notes='', tags=[], config=cfg, reinit=True)
        with run:
            #######################
            ### DATA GENERATION ###
            #######################

            # generate graph
            dag = data_gen.generate_dag(num_vars=cfg.graph.num_vars,
                                        edge_prob=cfg.graph.e_n / cfg.graph.num_vars,
                                        fns='linear gaussian',
                                        mu=cfg.dist.obs_mean,
                                        sigma=cfg.dist.obs_std,
                                        seed=seed)
            variables = [v.name for v in dag.variables]
            true_graph = dag.nx_graph
            mec = dag_to_mec(true_graph)

            # logging
            plot_graph(true_graph, "true graph")
            plot_graph(mec, "MEC")
            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)

            # datasets
            synth_dataset, interventions = data_gen.generate_data(dag=dag,
                                                                  n_obs=cfg.dist.n_obs,
                                                                  int_ratio=cfg.dist.int_ratio,
                                                                  seed=seed,
                                                                  int_mu=cfg.dist.int_mean,
                                                                  int_sigma=cfg.dist.int_std)

            target_dataset = data.PartitionData(features=synth_dataset.features[..., :-1],
                                                targets=synth_dataset.targets)
            target_dataset.update_partitions(target_dataset.targets)

            #########################
            ### ORACLE PC+CONTEXT ###
            #########################

            if cfg.do.oracle:
                pc_context_graph = add_context_vars(graph=true_graph.copy(),
                                                    n=cfg.oracle.n_int_targets,
                                                    vars=variables,
                                                    confounded=True)
                pc_context_mec = dag_to_mec(pc_context_graph)

                # logging
                plot_graph(pc_context_graph, "PC+context graph")
                plot_graph(pc_context_mec, "Oracle PC+context MEC")
                metrics.log_cd_metrics(true_graph, pc_context_mec, mec, "Oracle PC+context")

            ##################
            ### CLUSTERING ###
            ##################

            clusterer = instantiate(cfg.clustering.clusterer)
            clusterer.fit(synth_dataset.features[..., :-1])
            synth_dataset.memberships = clusterer.memberships_
            if cfg.clustering.name == "hdbscan_soft_normed":
                synth_dataset.memberships = synth_dataset.memberships / np.sum(synth_dataset.memberships, axis=1, keepdims=True)

            synth_dataset.update_partitions(clusterer.labels_)
            wandb.log({"cluster sizes": wandb.Histogram(clusterer.labels_)})
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(synth_dataset.targets, clusterer.labels_)
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(synth_dataset.targets, clusterer.labels_)

            ########################
            ### CAUSAL DISCOVERY ###
            ########################

            if cfg.do.causal_discovery:
                cd_model = instantiate(cfg.causal_discovery.model)
                cd_input = cd.prepare_data(cfg=cfg, data=synth_dataset, variables=variables)
                pred_graph = cd_model.predict(cd_input)
                context_graph = pred_graph.copy()

                # logging
                # tbl = wandb.Table(dataframe=df)
                # wandb.log({"clustered data": tbl})

                metrics.log_cd_metrics(true_graph, pred_graph, mec, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
                plot_graph(pred_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}")

            #########################
            ### CLUSTER DISCOVERY ###
            #########################

            if cfg.do.cluster_discovery:
                # Match pred and target clusters via the highest count
                int_targets = [i-1 for i in match_clusters(synth_dataset, target_dataset)]

                shds = []
                for i, cluster in enumerate(synth_dataset.partitions):
                    cd_model = instantiate(cfg.causal_discovery.model)
                    cluster_dataset = data.PartitionData(features=cluster.features[..., :-1])
                    df = cd.prepare_data(cfg=cfg, data=cluster_dataset, variables=variables)
                    pred_cluster_graph = cd_model.predict(df)
                    plot_graph(pred_cluster_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}, cluster {i}")

                    # interventional graph of the matched interventional distribution
                    true_int_graph = get_interventional_graph(true_graph, int_targets[i])
                    shds.append(cdt.metrics.SHD(true_int_graph, pred_cluster_graph, double_for_anticausal=False))

                wandb.run.summary["Pred clusters: SHD (avg)"] = np.mean(shds)

            ################
            ### ANALYSIS ###
            ################

            if cfg.do.context_analysis:
                if cfg.clustering.name == "target":
                    # FN edges from context variables to intervention targets
                    tps = 0
                    fps = 0
                    pred_adj_matrix = nx.to_numpy_array(context_graph)
                    for i in range(cfg.graph.num_vars):
                        tps += pred_adj_matrix[cfg.graph.num_vars + i + 1, i] == 1 # +1 because the first cluster (index 0) is observational
                        temp_matrix = pred_adj_matrix.copy()
                        temp_matrix[cfg.graph.num_vars + i + 1, i] = 0 # remove edge from context to target variable
                        temp_matrix[i, cfg.graph.num_vars + i + 1] = 0 # remove edge from target to context variable
                        fps += np.sum(temp_matrix[cfg.graph.num_vars + i + 1,:]) + np.sum(temp_matrix[:,cfg.graph.num_vars + i + 1])

                    tps /= cfg.graph.num_vars
                    fps /= cfg.graph.num_vars

                    wandb.run.summary["PC+context target: TPs context vars"] = tps
                    wandb.run.summary["PC+context target: FPs context vars"] = fps

                plot_graph(context_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}, context graph")

            wandb.finish()



if __name__ == '__main__':
    main()