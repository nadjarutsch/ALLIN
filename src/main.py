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

import sklearn
from clustering.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import random


os.environ['WANDB_MODE'] = 'offline'

# enable adding two variables in config file    TODO: is this used anywhere?
OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))

@hydra.main(config_path="./config", config_name="config")
def predict_graphs(cfg: DictConfig) -> float:
    """...
        Args:
            cfg: run configurations (e.g. hyperparameters) as specified in config-file and command line

        Returns:
            The SHD between the predicted graphs and the true graphs, averaged over all runs.
    """
    
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
            # each graph is entangled with a different distribution, according to the mode
            dags = []
            for mean in cfg.dist.obs_means:
                dag = data_gen.generate_dag(num_vars=cfg.graph.num_vars,
                                            edge_prob=cfg.graph.e_n / cfg.graph.num_vars,
                                            fns='linear gaussian',
                                            mu=mean,
                                            sigma=cfg.dist.obs_std,
                                            negative=cfg.dist.negative,
                                            seed=seed)
                dags.append(dag)

            # store graph properties
            variables = [v.name for v in dags[0].variables]
            true_graph = dags[0].nx_graph   # graphs are identical, only distributions differ
            mec = dag_to_mec(true_graph)

            # logging
            plot_graph(true_graph, "true graph")
            plot_graph(mec, "MEC")
            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)

            # datasets
            n_obs = int(cfg.dist.n / (1 + cfg.dist.int_ratio * cfg.n_int_targets))
            int_variables = random.sample(dags[0].variables, cfg.n_int_targets)
            datasets = []
            int_ratio = cfg.dist.int_ratio / len(cfg.dist.obs_means)
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

            #########################
            ### ORACLE PC+CONTEXT ###
            #########################

            if cfg.do.oracle:
                pc_context_graph = add_context_vars(graph=true_graph.copy(),
                                                    n=cfg.n_int_targets,
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

            # set number of clusters automatically if not manually specified (n_clusters == 0)
            if (cfg.clustering.name in ["K-Means", "GMM"]) and (cfg.clustering.clusterer.n_clusters == 0):
                cfg.clustering.clusterer.n_clusters = cfg.n_int_targets + 1

            clusterer = instantiate(cfg.clustering.clusterer)

            # pass knowledge of intervention targets and root nodes to the target clusterer
            if "Target" in cfg.clustering.name:
                clusterer.roots = [variables.index(n) for n in get_root_nodes(true_graph)]
                clusterer.int_targets = [variables.index(v.name) for v in int_variables]

            clusterer.fit(synth_dataset.features[..., :-1])

            synth_dataset.memberships = clusterer.memberships_
            synth_dataset.update_partitions(clusterer.labels_)
            wandb.log({"cluster sizes": wandb.Histogram(clusterer.labels_)})
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(synth_dataset.targets, clusterer.labels_)
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(synth_dataset.targets, clusterer.labels_)

            ########################
            ### CAUSAL DISCOVERY ###
            ########################

            n_clusters = clusterer.memberships_.shape[-1]
            if cfg.clustering.name != "Observational" and cfg.clustering.name != "None":
                try:
                    cfg.causal_discovery.model.mixture_model.n_input = n_clusters
                except:
                    pass

            cd_model = instantiate(cfg.causal_discovery.model)
            cd_input = cd.prepare_data(cfg=cfg, data=synth_dataset, variables=variables)
            pred_graph = cd_model.predict(cd_input)

            metrics.log_cd_metrics(true_graph, pred_graph, mec, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
            plot_graph(pred_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
            shds.append(wandb.run.summary["SHD"])

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
                    sns.kdeplot(data=obs_data, color=palette_obs, fill=True, label="observational").set(title=f"Marginal of {variables[i]}")
                    sns.kdeplot(data=int_data, color=palette_int, fill=True, label="interventional")

                    if cfg.do.causal_discovery and "IDIOD" in cfg.causal_discovery.name:
                        plt.axvline(wandb.run.summary[f"bias_obs_{i}"], 0, 1, color=palette_obs)
                        plt.axvline(wandb.run.summary[f"bias_int_{i}"], 0, 1, color=palette_int)

                    plt.legend()
                    plt.savefig(f"marginal_seed_{seed}_{variables[i]}.pdf")
                    plt.close()

            wandb.finish()

    return sum(shds) / len(shds)


if __name__ == '__main__':
    predict_graphs()
