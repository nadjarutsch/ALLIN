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
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import random
import json
import pickle

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['WANDB_MODE'] = 'offline'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))

sns.set_theme()
sns.set_context("paper")


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

    targets_list = []
    mse_losses_obs = []
    mse_losses_int = []
    nll_losses_obs = []
    nll_losses_int = []

    for seed in range(cfg.start_seed, cfg.end_seed):
        cfg.seed = seed
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

            synth_dataset = data.PartitionData(
                features=torch.cat([dataset.features[..., :-1] for dataset in datasets], dim=0),
                targets=np.concatenate([dataset.targets for dataset in datasets], axis=0))

            target_dataset = data.PartitionData(features=synth_dataset.features[..., :-1],
                                                targets=synth_dataset.targets)
            target_dataset.update_partitions(target_dataset.targets)

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


            ########################
            ### CAUSAL DISCOVERY ###
            ########################

            cd_model = instantiate(cfg.causal_discovery.model)
            cd_input = cd.prepare_data(cfg=cfg, data=synth_dataset, variables=variables)
            pred_graph = cd_model.predict(cd_input)

            metrics.log_cd_metrics(true_graph, pred_graph, mec,
                                   f"{cfg.causal_discovery.name} {cfg.clustering.name}")
            plot_graph(pred_graph, f"{cfg.causal_discovery.name} {cfg.clustering.name}")

            ################
            ### ANALYSIS ###
            ################

            targets = synth_dataset.features[..., :-1]
       #     intv = torch.repeat_interleave(input=torch.from_numpy(synth_dataset.targets != 0).int(), repeats=cfg.graph.num_vars)

            # interventions on variable i
            for i in range(cfg.graph.num_vars):
                intv = torch.from_numpy(synth_dataset.targets == i + 1).int()
                preds_obs = targets @ cd_model.W_est
                var_obs = 1 / (cfg.dist.n - cfg.graph.num_vars - 1) * torch.sum((targets - preds_obs)**2, dim=0)
                var_int = 1 / (cfg.dist.n - cfg.graph.num_vars - 1) * torch.sum(targets**2, dim=0)

                mse_obs = ((targets - preds_obs)**2)[:, i].flatten()
                mse_int = (targets**2)[:, i].flatten()

             #   mse_obs = ((targets - preds_obs) ** 2)
             #   mse_int = (targets**2)

                perm_mse = mse_obs.argsort(descending=True).numpy()

                nll = nn.GaussianNLLLoss(reduction='none')
                nll_obs = nll(preds_obs, targets, var_obs[None, i].repeat(targets.shape[0], 1))[:, i].flatten()
                nll_int = nll(torch.zeros_like(preds_obs), targets, var_int[None, i].repeat(targets.shape[0], 1))[:, i].flatten()

            #    nll_obs = nll(preds_obs, targets, var_obs[None, :].repeat(targets.shape[0], 1))
            #    nll_int = nll(torch.zeros_like(preds_obs), targets, var_int[None, :].repeat(targets.shape[0], 1))
                perm_nll = nll_obs.argsort(descending=True).numpy()

                legend_map = {0: "Observational",
                              1: "Interventional"}

                ax1 = sns.scatterplot(x=np.arange(0, mse_obs.shape[0]),
                                      y=mse_int[perm_mse].numpy(),
                                      hue=np.vectorize(legend_map.get)(intv[perm_mse].numpy()),
                                      hue_order=["Observational", "Interventional"],
                                      palette=sns.color_palette("Set2")[:2],
                                      linewidth=0,
                                      s=18)
                ax2 = sns.lineplot(data=mse_obs[perm_mse], color="black", label="Observational MSE")

                ax1.set(xticklabels=[])
                ax2.set(xticklabels=[])
                plt.legend()
                plt.ylabel("MSE")
                plt.title(f"Interventional MSE of variable {variables[i]}")
                plt.savefig(f"mse_{variables[i]}.pdf")

             #   plt.show()
                plt.close()

                ax1 = sns.scatterplot(x=np.arange(0, nll_obs.shape[0]),
                                y=nll_int[perm_nll].numpy(),
                                hue=np.vectorize(legend_map.get)(intv[perm_nll].numpy()),
                                hue_order=["Observational", "Interventional"],
                                palette=sns.color_palette("Set2")[:2],
                                linewidth=0,
                                s=18)
                ax2 = sns.lineplot(data=nll_obs[perm_nll], color="black", label="Observational NLL")

                ax1.set(xticklabels=[])
                ax2.set(xticklabels=[])
                plt.legend()
                plt.title(f"Interventional NLL of variable {variables[i]}")
                plt.ylabel("NLL")
                plt.savefig(f"nll_{variables[i]}.pdf")

             #   plt.show()
                plt.close()
    #    targets_list.append(targets)
    #    mse_losses_obs.append(mse_obs)
    #    mse_losses_int.append(mse_int)
    #    nll_losses_obs.append(nll_obs)
    #    nll_losses_int.append(nll_int)

   # for i, lst in enumerate([targets_list, mse_losses_obs, mse_losses_int, nll_losses_obs, nll_losses_int]):
   #     with open(f'data_{i}.pickle', 'wb') as handle:
   #         pickle.dump(lst, handle, protocol=pickle.HIGHEST_PROTOCOL)



  #  targets_json = json.dumps(targets_list)
  #  mse_obs_json = json.dumps(mse_losses_obs)
  #  mse_int_json = json.dumps(mse_losses_int)
  #  nll_obs_json = json.dumps(nll_losses_obs)
  #  nll_int_json = json.dumps(nll_losses_int)






if __name__ == '__main__':
    main()
