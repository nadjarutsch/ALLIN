import wandb
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import os
if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')

import networkx as nx
import cdt

import data_generation.data_generation as data_gen
import data_generation.datasets as data
import causal_discovery.prepare_data as cd
import metrics
from plotting import plot_graph
from data_generation.causal_graphs.graph_utils import dag_to_mec, add_context_vars, get_interventional_graph, get_root_nodes

import sklearn
from clustering.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import random


os.environ['WANDB_MODE'] = 'offline'

# enable adding two variables in config file
OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))

@hydra.main(config_path="./config", config_name="config")
def predict_graphs(cfg: DictConfig) -> float:
    """...
        Args:
            cfg:

        Returns:
            The SHD between the predicted graphs and the true graphs, averaged over all runs.
    """
    
    if torch.cuda.is_available():
        cfg.device = 'cuda:0'
#        try:
#            cfg.causal_discovery.model.device = 'cuda:0'
#        except:
#            pass
    else:
        cfg.device = 'cpu'

    shds = []

    print(cfg.causal_discovery.model.device)

    for seed in range(cfg.start_seed, cfg.end_seed):
        cfg.seed = seed
      #  if str(cfg.clustering.name) == "Target non-roots":
      #      cfg.clustering.clusterer.roots = None
        if str(cfg.clustering.name) == "K-Means":  # TODO: with resolver (hydra)
            cfg.clustering.clusterer.n_clusters = cfg.n_int_targets + 1
        if "GMM" in str(cfg.clustering.name):
            cfg.clustering.clusterer.n_components = cfg.n_int_targets + 1

        run = wandb.init(project=cfg.wandb.project,
                         entity=cfg.wandb.entity,
                         group=cfg.wandb.group,
                         notes='',
                         tags=[],
                         config=OmegaConf.to_container(cfg, resolve=True),
                         reinit=True)
        with run:
            #######################
            ### DATA GENERATION ###
            #######################

            # generate graph
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

            variables = [v.name for v in dags[0].variables]
            true_graph = dags[0].nx_graph
            if str(cfg.clustering.name) == "Target non-roots":
                cfg.clustering.clusterer.roots = [variables.index(n) for n in get_root_nodes(true_graph)]
            mec = dag_to_mec(true_graph)

            # logging
            plot_graph(true_graph, "true graph")
            plot_graph(mec, "MEC")
            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)
      #      wandb.run.summary["SHD zero guess"] = true_graph.size()

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

            if cfg.observational:
                synth_dataset = data.PartitionData(features=synth_dataset.features[:n_obs, :-1],   # does not work with multimodal
                                                   targets=synth_dataset.targets[:n_obs])

            clusterer = instantiate(cfg.clustering.clusterer)
            if isinstance(clusterer, TargetClusterer):
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

            if cfg.do.causal_discovery:
                n_clusters = clusterer.memberships_.shape[-1]
                if cfg.clustering.name != "Observational" and cfg.clustering.name != "None":
                    try:
                        cfg.causal_discovery.model.mixture_model.n_input = n_clusters
                    except:
                        pass

                if cfg.do.bootstrap:
                    pred_adj_matrix = np.zeros((cfg.graph.num_vars + n_clusters, cfg.graph.num_vars + n_clusters))
                    alpha_lst = [0.1, 0.05, 0.01, 0.005, 0.001]
                    for i in range(11):
                        # data bootstrapping
                        indices = np.random.choice(len(synth_dataset), size=int(99/100 * len(synth_dataset)), replace=False)
                        sub_dataset = data.PartitionData(features=synth_dataset.features[indices, :-1],
                                                         targets=synth_dataset.targets[indices])
                        sub_dataset.memberships = synth_dataset.memberships[indices]
                        sub_dataset.labels = synth_dataset.labels[indices]
                        cd_model = instantiate(cfg.causal_discovery.model)
                        cd_input = cd.prepare_data(cfg=cfg, data=sub_dataset, variables=variables)

                        # model bootstrapping
                        # cfg.causal_discovery.model.alpha = alpha_lst[i]
                        # cd_input = cd.prepare_data(cfg=cfg, data=synth_dataset, variables=variables)

                        pred_graph = cd_model.predict(cd_input)
                        pred_adj_matrix += nx.to_numpy_array(pred_graph)

                    pred_adj_matrix = np.round(pred_adj_matrix * 1/11)
                    mapping = dict(zip(range(len(variables)), variables))
                    pred_graph = nx.relabel_nodes(nx.DiGraph(incoming_graph_data=pred_adj_matrix), mapping)
                else:
                    cd_model = instantiate(cfg.causal_discovery.model)
                    cd_input = cd.prepare_data(cfg=cfg, data=synth_dataset, variables=variables)
                    pred_graph = cd_model.predict(cd_input)

                context_graph = pred_graph.copy()

                # logging
                # tbl = wandb.Table(dataframe=df)
                # wandb.log({"clustered data": tbl})

                metrics.log_cd_metrics(true_graph, pred_graph, mec, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
                plot_graph(pred_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}")
                shds.append(wandb.run.summary["SHD"])

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
                if cfg.clustering.name == "Target":
                    # FN edges from context variables to intervention targets
                    tps = 0
                    fps = 0
                    pred_adj_matrix = nx.to_numpy_array(context_graph)
                    for i in range(cfg.graph.num_vars):
                        tps += pred_adj_matrix[cfg.graph.num_vars + i + 1, i] == 1  # +1 because the first cluster (index 0) is observational
                        temp_matrix = pred_adj_matrix.copy()
                        temp_matrix[cfg.graph.num_vars + i + 1, i] = 0  # remove edge from context to target variable
                        temp_matrix[i, cfg.graph.num_vars + i + 1] = 0  # remove edge from target to context variable
                        fps += np.sum(temp_matrix[cfg.graph.num_vars + i + 1, :]) + np.sum(temp_matrix[:, cfg.graph.num_vars + i + 1])

                    tps /= cfg.graph.num_vars
                    fps /= cfg.graph.num_vars

                    wandb.run.summary["PC+context target: TPs context vars"] = tps
                    wandb.run.summary["PC+context target: FPs context vars"] = fps

                plot_graph(context_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}, context graph")

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

         #   if cfg.do.search_modes:
         #       clustering = sklearn.cluster.MeanShift().fit(synth_dataset.features[..., :-1])
         #       n_modes = len(np.unique(clustering.labels_))

            wandb.finish()

    return sum(shds) / len(shds)


if __name__ == '__main__':
    predict_graphs()
