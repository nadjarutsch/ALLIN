import networkx as nx
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generation.datasets import *
from causal_discovery.idiod.mixture import IdentityMixture
import os
import uuid
import numpy as np
import wandb
from itertools import chain
import shutil
import sklearn
from utils import set_seed
from typing import Union

from causal_discovery.notears_model import Notears
from causal_discovery.notears_model import NotearsAdv
from causal_discovery.notears.linear import allin_linear
from causal_discovery.notears.linear import allin_linear_adv


loss_dict = {'mse': nn.MSELoss(reduction='none')}

class IDIOD(nn.Module):
    def __init__(self,
                 d,
                 mixture_model,
                 lambda1,
                 loss_type="l2",
                 max_iter=100,
                 h_tol=1e-8,
                 rho_max=1e+16,
                 w_threshold=0.3,
                 batch_size=128,
                 max_epochs=10,
                 device='cpu',
                 patience=10,
                 lr=0.001,
                 max_steps=1,
                 name='idiod_new',
                 clustering='none',
                 apply_threshold=False,
                 single_target=False,
                 save_model=False,
                 log_progress=False,
                 save_w_est=True,
                 seed=-1,
                 deterministic=False,
                 obs_prior_prob=0.5,
                 fix_bias=False,
                 speedup=True,
                 auto_thresh=False,
                 alpha=0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter    # iterating over Lagrangian multipliers
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.batch_size = batch_size
        self.max_epochs = max_epochs    # unconstrained optimization
        self.lr = lr
        self.d = d
        self.device = device
        self.max_steps = max_steps   # alternating between learning assignments and obs / int models
        self.step = 0   # for logging
        self.single_target = single_target
        self.save_model = save_model
        self.log_progress = log_progress
        self.save_w_est = save_w_est
        self.name = name
        self.clustering = clustering
        self.seed = seed
        self.deterministic = deterministic
        self.obs_prior_prob = obs_prior_prob
        self.fix_bias = fix_bias
        self.speedup = speedup
        self.auto_thresh = auto_thresh
        self.alpha = alpha

        # models
        self.loss = loss_dict['mse']
        self.model_obs = LinearFixedParams(in_features=self.d,
                                           out_features=self.d,
                                           mask=torch.ones((self.d, self.d), dtype=torch.bool).fill_diagonal_(0),
                                           fixed_params=torch.zeros((self.d, self.d)),
                                           device=self.device)

        self.model_int = nn.Linear(in_features=self.d,
                                   out_features=self.d,
                                   device=self.device)

        self.mixture = mixture_model.to(self.device)
        self.model_obs_threshold = LinearThreshold(self.model_obs, self.w_threshold)

        self.mix_coeff_obs = obs_prior_prob

        # init params
        set_seed(seed)
        for i, param in enumerate(self.mixture.parameters()):
            if len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.constant_(param, 0)

        # init classification bias with prior probability of the observational regime
        if not isinstance(self.mixture, IdentityMixture):
            nn.init.constant_(self.mixture.layers[-2].bias, np.log(self.obs_prior_prob / (1 - self.obs_prior_prob)))

        for param in chain(self.model_obs.parameters(), self.model_int.parameters()):
            nn.init.constant_(param, 0)

        # freeze weights
        self.model_obs.bias.requires_grad = False   # for pretraining
        self.model_int.weight.requires_grad = False
        if not isinstance(self.mixture, IdentityMixture):
            self.mixture.layers[-2].bias.requires_grad = not self.fix_bias

        # early stopping
        self.path = os.path.join('causal_discovery', name, 'saved_models', str(uuid.uuid1()))
        os.makedirs(self.path)
        self.patience = patience

        # clustering (optional)
        self.clustering = clustering
        self.p = None

        self.apply_threshold = not self.auto_thresh

    def predict(self, cd_input: tuple):
        variables, data = cd_input
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=not self.deterministic)
        optimizer_obs = optim.Adam(self.model_obs.parameters(), lr=self.lr)
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff

        # pretrain
        print("\n Starting pretraining...")
        if self.speedup:
            if self.auto_thresh:
                notears = NotearsAdv(self.lambda1, self.loss_type, self.max_iter, self.h_tol, self.rho_max, self.alpha)
            else:
                notears = Notears(self.lambda1, self.loss_type, self.max_iter, self.h_tol, self.rho_max, self.w_threshold)
            notears_in = (variables, data.tensors[0].clone().numpy())
            notears.predict(notears_in)
            self.model_obs.weight.data.copy_(
                torch.from_numpy(notears.W_est.T))  # TODO: make un-thresholded version available
        else:
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_obs],
                                                     mixture=False)

        self.model_obs.bias.requires_grad = True

        all_feats = data.tensors[0].to(self.device)
        preds = self.model_obs(all_feats)
        targets = data.tensors[0].to(self.device)
        k = round(all_feats.shape[0] * (1-self.mix_coeff_obs)) * all_feats.shape[1]
        losses = self.loss(preds, targets)
        indices_high_losses = torch.topk(self.loss(preds, targets).flatten(), k).indices

        features_flat = all_feats.clone()
        features_flat = features_flat.flatten()

        probs = torch.ones_like(features_flat, requires_grad=False)
        probs[indices_high_losses] = 0
        probs = probs.reshape_as(all_feats)

        for _ in range(self.max_steps):
            # learn distribution assignments
            print("\n Searching for interventional data...")
          #  mu_intv = torch.sum(probs * data.tensors[0], dim=0) / torch.sum(probs, dim=0)
          #  mu_obs = torch.sum((1 - probs) * data.tensors[0], dim=0) / torch.sum(1 - probs, dim=0)
            means_intv = self.model_int(all_feats)
            means_obs = self.model_obs_threshold(all_feats)

            vars_intv = torch.sum((1 - probs) * (all_feats - means_intv)**2, dim=0) / torch.sum(1 - probs, dim=0)
            vars_obs = torch.sum(probs * (all_feats - means_obs)**2, dim=0) / torch.sum(probs, dim=0)

            self.mix_coeff_obs = torch.mean(probs, dim=0)
            ll_intv = torch.exp(- 1/2 * (all_feats - means_intv)**2 / vars_intv) / torch.sqrt(vars_intv)
            ll_obs = torch.exp(- 1/2 * (all_feats - means_obs)**2 / vars_obs) / torch.sqrt(vars_obs)

            probs = (self.mix_coeff_obs * ll_obs / (self.mix_coeff_obs * ll_obs + (1 - self.mix_coeff_obs) * ll_intv)).detach()



         #   probs = self.mixture(data.tensors[1].to(self.device)).clone().detach().cpu().numpy()
       #     if not isinstance(self.mixture, IdentityMixture):
       #         optimizer_mix = optim.Adam(self.mixture.parameters(), lr=self.lr)
       #         rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
       #                                                  rho=rho,
        #                                                 alpha=alpha,
         #                                                h=h,
          #                                               optimizers=[optimizer_mix],
           #                                              mixture=True,
            #                                             apply_threshold=self.apply_threshold)

            # relearn weights
            print("\n Adjusting weights...")

            notears_in = data.tensors[0].clone().numpy()


            if self.auto_thresh:
                W_est = allin_linear_adv(X=notears_in,
                                     P=probs.numpy(),
                                     lambda1=self.lambda1,
                                     loss_type=self.loss_type,
                                     max_iter=self.max_iter,
                                     h_tol=self.h_tol,
                                     rho_max=self.rho_max,
                                     thresh=self.alpha)
            else:
                W_est = allin_linear(X=notears_in,
                                     P=probs.numpy(),
                                     lambda1=self.lambda1,
                                     loss_type=self.loss_type,
                                     max_iter=self.max_iter,
                                     h_tol=self.h_tol,
                                     rho_max=self.rho_max,
                                     w_threshold=self.w_threshold)

                self.W_est = W_est

            W_obs_augmented, W_int_augmented = np.split(W_est, 2, axis=0)
            W_obs = W_obs_augmented[:-1, ...]
            bias_obs = W_obs_augmented[-1, ...]
            bias_int = W_int_augmented[-1, ...]
            self.model_obs.weight.data.copy_(torch.from_numpy(W_obs.T))
            self.model_obs.bias.data.copy_(torch.from_numpy(bias_obs).squeeze())
            self.model_int.bias.data.copy_(torch.from_numpy(bias_int).squeeze())

        W_est = self.model_obs.weight[:self.d, ...].detach().cpu().numpy().T
        if self.save_w_est:
            np.savetxt(f'{self.name}_{self.clustering}_seed_{self.seed}.txt', W_est)
        if not self.auto_thresh:
            W_est[np.abs(W_est) < self.w_threshold] = 0
        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        if not self.save_model:
            shutil.rmtree(self.path)

        eval_dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        labels = []
        dist_keys = ["obs"] + variables
        p_correct = np.zeros(len(variables) + 1)
        n_int_assign = 0
        n_int_correct = 0
        for batch in eval_dataloader:
            features, mixture_in, targets = batch
            mixture_in = mixture_in.to(self.device)
            features = features.to(self.device)
        #    probs = self.mixture(mixture_in)
            means_intv = self.model_int(features)
            means_obs = self.model_obs_threshold(features)

       #     vars_intv = torch.sum((1 - probs) * (features - means_intv) ** 2, dim=0) / torch.sum(1 - probs, dim=0)
       #     vars_obs = torch.sum(probs * (features - means_obs) ** 2, dim=0) / torch.sum(probs, dim=0)

            ll_intv = torch.exp(- 1 / 2 * (features - means_intv) ** 2 / vars_intv) / torch.sqrt(vars_intv)
            ll_obs = torch.exp(- 1 / 2 * (features - means_obs) ** 2 / vars_obs) / torch.sqrt(vars_obs)

            probs = (self.mix_coeff_obs * ll_obs / (self.mix_coeff_obs * ll_obs + (1 - self.mix_coeff_obs) * ll_intv)).detach()

            assignments = torch.round(probs)
            labels_batch = torch.sum(assignments * (2 ** torch.tensor(list(range(len(variables))), device=self.device)),
                                     dim=1).squeeze().tolist()
            labels.extend(labels_batch)

            # for precision & recall
            n_int_assign += torch.sum(1 - assignments).item()
            n_int_correct += np.sum([torch.sum(assignments[targets == i+1][..., i] == 0).item() for i in range(len(variables))])

            probs = probs.detach().cpu().numpy()
            targets_int = np.copy(targets)
            targets_int[targets_int > 0] = targets_int[targets_int > 0] - 1
            int_probs = 1 - probs[np.arange(targets.shape[0])[:, None], targets_int[:, None]]
            obs_probs = np.mean(probs[targets == 0], axis=1)
            p_correct_batch = int_probs.squeeze()
            p_correct_batch[targets == 0] = obs_probs
            for target in set(targets.tolist()):
                p_corr = np.sum(p_correct_batch, where=(targets == target))
                p_correct[target] += p_corr

        counts = np.array([data.tensors[2].tolist().count(i) for i in range(len(variables) + 1)])
        p_correct = (p_correct / counts).tolist()

        n_int = np.count_nonzero(data.tensors[2])
        wandb.run.summary["Precision intv"] = n_int_correct / n_int_assign
        wandb.run.summary["Recall intv"] = n_int_correct / n_int

        for i, p in enumerate(p_correct):
            wandb.run.summary[f"p_{dist_keys[i]}"] = p
        try:
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(data.tensors[2], labels)
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(data.tensors[2], labels)
        except:
            pass
        wandb.run.summary["n_clusters"] = len(set(labels))
        bias_list_obs = self.model_obs.bias.detach().cpu().tolist()
        bias_list_int = self.model_int.bias.detach().cpu().tolist()
        for i, (bias_obs, bias_int) in enumerate(zip(bias_list_obs, bias_list_int)):
            wandb.run.summary[f"bias_obs_{variables[i]}"] = bias_obs
            wandb.run.summary[f"bias_int_{variables[i]}"] = bias_int

        return nx.relabel_nodes(pred_graph, mapping)

    def optimize_lagrangian(self, dataloader, rho, alpha, h, optimizers, mixture, apply_threshold=False):
        self.eval()
        params_init = [self.model_obs.weight.detach().clone(),
                       self.model_obs.bias.detach().clone(),
                       self.model_int.weight.detach().clone(),
                       self.model_int.bias.detach().clone()]
        for _ in range(self.max_iter):
            h_new = None
            while rho < self.rho_max:
                self.optimize(dataloader, rho, h, alpha, optimizers, mixture, params_init, apply_threshold)
                h_new = self._h(self.model_obs.weight)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break

            h = h_new.detach()
            params_init = [self.model_obs.weight.detach().clone(),
                           self.model_obs.bias.detach().clone(),
                           self.model_int.weight.detach().clone(),
                           self.model_int.bias.detach().clone()]
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                return rho, alpha, h

    def optimize(self, dataloader, rho, h, alpha, optimizers, mixture, params_init, apply_threshold):
        # init params
        for param, init in zip(chain(self.model_obs.parameters(), self.model_int.parameters()), params_init):
            param.data.copy_(init)

        train_losses = []
        best_epoch, stop_count = 0, 0

        for epoch in range(self.max_epochs):
            self.train()

            for _, batch in enumerate(dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs = self.model_obs_threshold(features) if apply_threshold else self.model_obs(features)
                loss = self.loss(features, preds_obs)

                if mixture:
                    preds_int = self.model_int(features)
                    loss_int = self.loss(features, preds_int)
                    probs = self.mixture(mixture_in)

                    loss = probs * loss + (1 - probs) * loss_int

                    if self.log_progress:
                        wandb.log({'p_obs': torch.mean(probs)}, step=self.step)
                        names_obs = iter(["bias_obs_" + str(i) for i in range(len(self.model_obs.bias))])
                        names_int = iter(["bias_int_" + str(i) for i in range(len(self.model_int.bias))])
                        bias_obs_dict = dict(zip(names_obs, self.model_obs.bias.detach().cpu().tolist()))
                        bias_int_dict = dict(zip(names_int, self.model_int.bias.detach().cpu().tolist()))
                        wandb.log(bias_obs_dict, step=self.step)
                        wandb.log(bias_int_dict, step=self.step)

                loss = 0.5 / features.shape[0] * torch.sum(loss)  # equivalent to original numpy implementation
                h = self._h(self.model_obs.weight)
                obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.model_obs.weight))
                obj.backward()

                if self.log_progress:
                    wandb.log({'loss (unconstrained)': loss}, step=self.step)
                    wandb.log({'loss (Lagrange)': obj}, step=self.step)
                self.step += 1

                for optimizer in optimizers:
                    optimizer.step()

            # evaluate performance on whole dataset (always without thresholding W_est) # TODO: try thresholding
            self.eval()
            loss_all = 0
            for _, batch in enumerate(dataloader):
                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                # preds_obs = self.model_obs_threshold(features) if apply_threshold else self.model_obs(features)   # TODO: use thresholding?
                preds_obs = self.model_obs(features)
                loss = self.loss(features, preds_obs)

                if mixture:
                    preds_int = self.model_int(features)
                    loss_int = self.loss(features, preds_int)
                    probs = self.mixture(mixture_in)
                    loss = probs * loss + (1 - probs) * loss_int

                loss_all += torch.sum(loss)

            h = self._h(self.model_obs.weight)
            obj = 0.5 / len(dataloader.dataset) * loss_all + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.model_obs.weight))
            train_losses.append(obj)
            print(f"[Epoch {epoch + 1:2d}] Training loss: {obj:05.5f}")

            if len(train_losses) == 1 or obj < train_losses[best_epoch]:
                print("\t   (New best performance, saving model...)")
                torch.save(self.state_dict(), os.path.join(self.path, 'model.pt'))
                best_epoch = epoch
                stop_count = 0
            else:
                stop_count += 1

            if stop_count >= self.patience:
                break

        self.load_state_dict(torch.load(os.path.join(self.path, 'model.pt')))

    def _h(self, W):
        """Evaluate value of acyclicity constraint."""
        E = torch.linalg.matrix_exp(W * W)  # (Zheng et al. 2018)
        h = torch.trace(E) - self.d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        return h


class LinearFixedParams(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 mask: torch.Tensor,
                 fixed_params: torch.Tensor,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super(LinearFixedParams, self).__init__(in_features, out_features, bias, device, dtype)
        mask = mask.to(device)
        self.register_buffer('mask', mask)
        fixed_params = fixed_params.to(device)
        self.register_buffer('fixed', fixed_params)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.where(self.mask, self.weight, self.fixed)
        return F.linear(input, weight, self.bias)


class LinearThreshold(nn.Module):
    def __init__(self, layer: Union[nn.Linear, LinearFixedParams], threshold: float):
        super().__init__()
        self.layer = layer
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(self.layer, LinearFixedParams):
            weight = torch.where(self.layer.mask, self.layer.weight, self.layer.fixed)
        else:
            weight = self.layer.weight.clone()

        weight[torch.abs(self.layer.weight) < self.threshold] = 0
        return F.linear(input, weight, self.layer.bias)


