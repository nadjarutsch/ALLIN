import wandb
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import os

if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')

import networkx as nx
# import causaldag
import cdt

import data_generation.data_generation as data_gen
import data_generation.datasets as data
import causal_discovery.prepare_data as cd
import metrics
from plotting import plot_graph
from data_generation.causal_graphs.graph_utils import dag_to_mec, add_context_vars, get_interventional_graph, \
    get_root_nodes

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from clustering.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['WANDB_MODE'] = 'offline'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    if torch.cuda.is_available():
        cdt.SETTINGS.rpath = '/sw/arch/Debian10/EB_production/2021/software/R/4.1.0-foss-2021a/lib/R/bin/Rscript'
        cfg.device = 'cuda:0'
        try:
            cfg.causal_discovery.model.device = 'cuda:0'
        except:
            pass
    else:
        cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'
        cfg.device = 'cpu'

    shds = []
    for seed in range(cfg.start_seed, cfg.end_seed):
        cfg.seed = seed
        if str(cfg.clustering.name) == "target_non_roots":
            cfg.clustering.clusterer.roots = None
        if str(cfg.clustering.name) == "kmeans":  # TODO: with resolver (hydra)
            cfg.clustering.clusterer.n_clusters = cfg.graph.num_vars + 1
        if "gmm" in str(cfg.clustering.name):
            cfg.clustering.clusterer.n_components = cfg.graph.num_vars + 1

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
            if str(cfg.clustering.name) == "target_non_roots":
                cfg.clustering.clusterer.roots = [variables.index(n) for n in get_root_nodes(true_graph)]
            mec = dag_to_mec(true_graph)

            # logging
            plot_graph(true_graph, "true graph")
            plot_graph(mec, "MEC")
            wandb.run.summary["avg neighbourhood size"] = metrics.avg_neighbourhood_size(dag)
    #      wandb.run.summary["SHD zero guess"] = true_graph.size()

    # datasets





if __name__ == '__main__':
    main()