import wandb
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')

import networkx as nx
import cdt

from utils import set_seed, get_device
import data_generation.data_generation as data_gen
import data_generation.datasets as data
import causal_discovery.prepare_data as cd
import metrics
from plotting import plot_graph
from data_generation.causal_graphs.graph_utils import dag_to_mec, add_context_vars, get_root_nodes
from data_generation.causal_graphs.graph_generation import generate_continuous_graph, generate_random_dag

import sklearn
from clustering.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import random


@hydra.main(config_path="./config", config_name="config")
def predict_graphs(cfg: DictConfig) -> float:
    """ Executes a causal discovery algorithm on synthetic graph data, sampled from a specified number of random graphs.
        The following causal discovery algorithms are supported:

            - PC
            - PC-JCI with Clustering (K-means, GMM)
            - PC-JCI oracle
            - NOTEARS
            - ALLIN (MSE, Gaussian)
            - ALLIN with Clustering
            - ALLIN K-oracle
            - ALLIN Z-oracle
            - DCDI-L
            - ENCO

        Args:
            cfg: run configurations (e.g. hyperparameters) as specified in config-file and command line

        Returns:
            The SHD between the predicted graphs and the true graphs, averaged over all runs.
    """

    os.environ['WANDB_MODE'] = 'offline'    # TODO: pass over from cfg
    cfg.device = get_device()   # cpu or gpu
    shds = []

    for seed in range(cfg.start_seed, cfg.end_seed):
        cfg.seed = seed

        run = wandb.init(project=cfg.wandb.project,
                         entity=cfg.wandb.entity,
                         group=cfg.wandb.group,
                         config=OmegaConf.to_container(cfg, resolve=True),
                         reinit=True)
        with run:
            #######################
            ### DATA GENERATION ###
            #######################

            # if the distribution is multimodal, we create the same graph for each mode
            # each graph is specified with a different distribution that represents its mode
            dags = []
            for mean in cfg.dist.obs_means:
                dag = generate_continuous_graph(num_vars=cfg.graph.num_vars,
                                                graph_func=generate_random_dag,
                                                p=cfg.graph.e_n / cfg.graph.num_vars,   # edge probability
                                                mu=mean,
                                                sigma=cfg.dist.obs_std,
                                                seed=seed)

                dags.append(dag)

            # store graph properties
            variable_names = [v.name for v in dags[0].variables]
            true_graph = dags[0].nx_graph   # graphs are identical, only distributions differ
            mec = dag_to_mec(true_graph)

            # logging (graph properties)
            plot_graph(true_graph, "true graph")
            plot_graph(mec, "MEC")
            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)

            # create datasets
            n_obs = int(cfg.dist.n / (1 + cfg.dist.int_ratio * cfg.n_int_targets))
            set_seed(cfg.seed)
            int_variables = random.sample(dags[0].variables, cfg.n_int_targets)     # sample intervened variables
            datasets = []
            int_ratio = cfg.dist.int_ratio / len(cfg.dist.obs_means)

            # for multimodal distributions, sample data from each mode as specified in the different DAG objects
            for dag in dags:
                synth_dataset, interventions = data_gen.generate_data(dag=dag,
                                                                      n_obs=n_obs,
                                                                      int_ratio=int_ratio,
                                                                      seed=seed,
                                                                      int_mu=cfg.dist.int_mean,
                                                                      int_sigma=cfg.dist.int_std,
                                                                      int_variables=int_variables)

                datasets.append(synth_dataset)

            synth_dataset = data.PartitionData(features=torch.cat([dataset.features[..., :-1] for dataset in datasets], dim=0),
                                               targets=np.concatenate([dataset.targets for dataset in datasets], axis=0))

            target_dataset = data.PartitionData(features=synth_dataset.features[..., :-1],
                                                targets=synth_dataset.targets)
            target_dataset.update_partitions(target_dataset.targets)

            ##################
            ### CLUSTERING ###
            ##################

            # set number of clusters automatically if not manually specified (n_clusters == 0)
            if (cfg.clustering.name in ["K-Means", "GMM"]) and (cfg.clustering.clusterer.n_clusters == 0):
                cfg.clustering.clusterer.n_clusters = cfg.n_int_targets + 1

            clusterer = instantiate(cfg.clustering.clusterer)

            # pass knowledge of intervention targets and root nodes to the oracle clusterer
            if "Target" in cfg.clustering.name:
                clusterer.roots = [variable_names.index(n) for n in get_root_nodes(true_graph)]
                clusterer.int_targets = [variable_names.index(v.name) for v in int_variables]

            clusterer.fit(synth_dataset.features[..., :-1])
            synth_dataset.memberships = clusterer.memberships_
            synth_dataset.update_partitions(clusterer.labels_)

            # logging (clustering performance)
            wandb.log({"cluster sizes": wandb.Histogram(clusterer.labels_)})
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(synth_dataset.targets, clusterer.labels_)
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(synth_dataset.targets, clusterer.labels_)

            ########################
            ### CAUSAL DISCOVERY ###
            ########################

            cd_model = instantiate(cfg.causal_discovery.model)
            cd_input = cd.prepare_data(cfg=cfg, data=synth_dataset, variables=variable_names)
            pred_graph = cd_model.predict(cd_input)

            # logging (causal discovery performance)
            metrics.log_cd_metrics(true_graph, pred_graph, mec, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
            plot_graph(pred_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
            shds.append(wandb.run.summary["SHD"])

            # TODO: move to ALLIN training
            ''' 
            ################
            ### ANALYSIS ###
            ################

            if cfg.do.plot_marginals:
                for i in range(cfg.graph.num_vars):
                    mask = np.ones(len(synth_dataset.features), dtype=bool)
                    mask[list(range(n_obs + i * n_obs * cfg.dist.int_ratio, n_obs + (i + 1) * n_obs * cfg.dist.int_ratio))] = False
                    obs_data = synth_dataset.features[..., i][mask]
                    obs_data = obs_data - torch.mean(synth_dataset.features[..., i])
                    int_data = synth_dataset.features[..., i][~mask]
                    int_data = int_data - torch.mean(synth_dataset.features[..., i])
                    palette_obs = matplotlib.colors.hex2color("#ef476f")
                    palette_int = matplotlib.colors.hex2color("#06d6a0")
                    sns.kdeplot(data=obs_data, color=palette_obs, fill=True, label="observational").set(title=f"Marginal of {variable_names[i]}")
                    sns.kdeplot(data=int_data, color=palette_int, fill=True, label="interventional")

                    if cfg.do.causal_discovery and "IDIOD" in cfg.causal_discovery.name:
                        plt.axvline(wandb.run.summary[f"bias_obs_{i}"], 0, 1, color=palette_obs)
                        plt.axvline(wandb.run.summary[f"bias_int_{i}"], 0, 1, color=palette_int)

                    plt.legend()
                    plt.savefig(f"marginal_seed_{seed}_{variable_names[i]}.pdf")
                    plt.close()
            '''

            wandb.finish()

    return sum(shds) / len(shds)


if __name__ == '__main__':
    predict_graphs()
