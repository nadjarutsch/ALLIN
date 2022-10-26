from causal_discovery.enco.causal_discovery.distribution_fitting import DistributionFitting
from causal_discovery.enco.causal_discovery.utils import track, find_best_acyclic_graph
from causal_discovery.enco.causal_discovery.multivariable_mlp import create_model
from causal_discovery.enco.causal_discovery.multivariable_flow import create_continuous_model
from causal_discovery.enco.causal_discovery.graph_fitting import GraphFitting
from causal_discovery.enco.causal_discovery.datasets import ObservationalCategoricalData
from causal_discovery.enco.causal_discovery.optimizers import AdamTheta, AdamGamma

import torch
import torch.nn as nn
import torch.utils.data as data
import networkx as nx

from omegaconf import DictConfig, OmegaConf

import numpy as np
import time
import sys
sys.path.append("../")  # TODO: needed?


class ENCO:
    def __init__(self,
                 num_vars,
                 hidden_dims=[],
                 use_flow_model=False,
                 lr_model=5e-3,
                 betas_model=(0.9, 0.999),
                 weight_decay=0.0,
                 lr_gamma=2e-2,
                 betas_gamma=(0.9, 0.9),
                 lr_theta=1e-1,
                 betas_theta=(0.9, 0.999),
                 model_iters=1000,
                 graph_iters=100,
                 batch_size=128,
                 GF_num_batches=1,
                 GF_num_graphs=100,
                 lambda_sparse=0.004,
                 latent_threshold=0.35,
                 use_theta_only_stage=False,
                 theta_only_num_graphs=4,
                 theta_only_iters=1000,
                 max_graph_stacking=200,
                 sample_size_obs=5000,
                 sample_size_inters=200,
                 num_epochs=30):

        self.num_vars = num_vars
        self.hidden_dims = OmegaConf.to_container(hidden_dims)
        self.use_flow_model = use_flow_model
        self.lr_model = lr_model
        self.betas_model = betas_model
        self.weight_decay = weight_decay
        self.lr_gamma = lr_gamma
        self.betas_gamma = betas_gamma
        self.lr_theta = lr_theta
        self.betas_theta = betas_theta
        self.model_iters = model_iters
        self.graph_iters = graph_iters
        self.batch_size = batch_size
        self.GF_num_batches = GF_num_batches
        self.GF_num_graphs = GF_num_graphs
        self.lambda_sparse = lambda_sparse
        self.latent_threshold = latent_threshold
        self.use_theta_only_stage = use_theta_only_stage
        self.theta_only_num_graphs = theta_only_num_graphs
        self.theta_only_iters = theta_only_iters
        self.max_graph_stacking = max_graph_stacking
        self.sample_size_obs = sample_size_obs
        self.sample_size_inters = sample_size_inters
        self.num_epochs = num_epochs

        # Initialize graph parameters
        self.init_graph_params()

    def predict(self, cd_input: tuple):

        variables, obs_dataset, int_dataset = cd_input

        model = create_continuous_model(num_vars=len(variables),
                                        hidden_dims=self.hidden_dims,
                                        use_flow_model=self.use_flow_model)

        model_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=self.lr_model,
                                           betas=self.betas_model,
                                           weight_decay=self.weight_decay)

        obs_dataloader = data.DataLoader(obs_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.distribution_fitting_module = DistributionFitting(model=model,
                                                               optimizer=model_optimizer,
                                                               data_loader=obs_dataloader)

        self.graph_fitting_module = GraphFitting(model=model,
                                                 num_vars=self.num_vars,
                                                 num_batches=self.GF_num_batches,
                                                 num_graphs=self.GF_num_graphs,
                                                 theta_only_num_graphs=self.theta_only_num_graphs,
                                                 batch_size=self.batch_size,
                                                 lambda_sparse=self.lambda_sparse,
                                                 max_graph_stacking=self.max_graph_stacking,
                                                 sample_size_inters=self.sample_size_inters,
                                                 dataset=int_dataset)

        pred_adjmatrix = self.discover_graph().numpy()  # TODO: potentially transpose?
        pred_graph = nx.from_numpy_array(pred_adjmatrix, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)

    def init_graph_params(self):
        """
        Initializes gamma and theta parameters, including their optimizers.
        """
        self.gamma = nn.Parameter(torch.zeros(self.num_vars, self.num_vars))  # Init with zero => prob 0.5
        self.gamma.data[torch.arange(self.num_vars), torch.arange(self.num_vars)] = -9e15  # Mask diagonal
        self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=self.lr_gamma, betas=self.betas_gamma)

        self.theta = nn.Parameter(torch.zeros(self.num_vars, self.num_vars))  # Init with zero => prob 0.5
        self.theta_optimizer = AdamTheta(self.theta, lr=self.lr_theta, beta1=self.betas_theta[0], beta2=self.betas_theta[1])

    def discover_graph(self, num_epochs=30, stop_early=False):
        """
        Main training function. It starts the loop of distribution and graph fitting.
        Returns the predicted binary adjacency matrix.
        """
        for epoch in track(range(num_epochs), leave=False, desc="Epoch loop"):
            self.epoch = epoch
            start_time = time.time()
            # Update Model
            self.distribution_fitting_step()
            self.dist_fit_time = time.time() - start_time
            # Update graph parameters
            self.graph_fitting_step()
            self.iter_time = time.time() - start_time

        return self.get_binary_adjmatrix()

    def distribution_fitting_step(self):
        """
        Performs on iteration of distribution fitting.
        """
        # Probabilities to sample input masks from
        sample_matrix = torch.sigmoid(self.gamma) * torch.sigmoid(self.theta)
        # Update model in a loop
        t = track(range(self.model_iters), leave=False, desc="Distribution fitting loop")
        for _ in t:
            loss = self.distribution_fitting_module.perform_update_step(sample_matrix=sample_matrix)
            if hasattr(t, "set_description"):
                t.set_description("Model update loop, loss: %4.2f" % loss)

    def graph_fitting_step(self):
        """
        Performs on iteration of graph fitting.
        """
        # For large graphs, freeze gamma in every second graph fitting stage
        only_theta = (self.use_theta_only_stage and self.epoch % 2 == 0)
        iters = self.graph_iters if not only_theta else self.theta_only_iters
        # Update gamma and theta in a loop
        for _ in track(range(iters), leave=False, desc="Graph fitting loop"):
            self.gamma_optimizer.zero_grad()
            self.theta_optimizer.zero_grad()
            theta_mask, var_idx = self.graph_fitting_module.perform_update_step(self.gamma,             # TODO: possible with var_idx as vector (i.e. different interventions)?
                                                                                self.theta,
                                                                                only_theta=only_theta)
            if not only_theta:  # In the gamma freezing stages, we do not update gamma
                if isinstance(self.gamma_optimizer, AdamGamma):
                    self.gamma_optimizer.step(var_idx)
                else:
                    self.gamma_optimizer.step()
            self.theta_optimizer.step(theta_mask)

    def get_binary_adjmatrix(self):
        """
        Returns the predicted, binary adjacency matrix of the causal graph.
        """
        binary_gamma = self.gamma > 0.0
        binary_theta = self.theta > 0.0
        A = binary_gamma * binary_theta

        return (A == 1).cpu()
