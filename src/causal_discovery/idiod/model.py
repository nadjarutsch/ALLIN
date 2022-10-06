import networkx as nx
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generation.datasets import *
from causal_discovery.idiod.mixture import IdentityMixture
from tqdm import tqdm
import os
import uuid
from models.mlp import MLP
import wandb
from itertools import chain
import shutil
import sklearn


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
                 relearn_iter=1,
                 name='idiod',
                 clustering='none',
                 apply_threshold=False,
                 loss='mse',
                 single_target=False,
                 save_model=False,
                 log_progress=False):
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
        self.relearn_iter = relearn_iter    # alternating between learning assignments and obs / int models
        self.step = 0   # for logging
        self.single_target = single_target
        self.save_model = save_model
        self.log_progress = log_progress

        # models
        self.loss = loss_dict[loss]
        if loss == 'likelihood':
            mask = torch.cat([torch.ones((self.d, self.d), dtype=torch.bool).fill_diagonal_(0),
                              torch.zeros((self.d, self.d), dtype=torch.bool)],
                             dim=0)
            self.model_obs = LinearFixedParams(in_features=self.d,
                                               out_features=self.d * 2,
                                               mask=mask,
                                               fixed_params=torch.zeros((self.d * 2, self.d)),
                                               device=self.device)
        elif loss == 'mse':
            self.model_obs = LinearFixedParams(in_features=self.d,
                                               out_features=self.d,
                                               mask=torch.ones((self.d, self.d), dtype=torch.bool).fill_diagonal_(0),
                                               fixed_params=torch.zeros((self.d, self.d)),
                                               device=self.device)

        self.model_int = nn.Linear(in_features=self.d,
                                   out_features=self.d * 2 if loss == 'likelihood' else self.d,
                                   device=self.device)

        self.mixture = mixture_model.to(self.device)
        self.model_obs_threshold = LinearThreshold(self.model_obs, self.w_threshold)

        # init params
        for param in chain(self.model_obs.parameters(), self.model_int.parameters()):#, [self.mixture.layers[-2].bias]):
            nn.init.constant_(param, 0)

        # freeze weights
        self.model_obs.bias.requires_grad = False   # for pretraining
        self.model_int.weight.requires_grad = False

        # early stopping
        self.path = os.path.join('causal_discovery', name, 'saved_models', str(uuid.uuid1()))
        os.makedirs(self.path)
        self.patience = patience

        # clustering (optional)
        self.clustering = clustering
        self.p = None

        self.apply_threshold = apply_threshold

    def predict(self, cd_input: tuple):
        variables, data = cd_input
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer_obs = optim.Adam(self.model_obs.parameters(), lr=self.lr)
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff

        # pretrain
        print("\n Starting pretraining...")
        rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                 rho=rho,
                                                 alpha=alpha,
                                                 h=h,
                                                 optimizers=[optimizer_obs],
                                                 mixture=False)

        self.model_obs.bias.requires_grad = True

        for _ in range(self.relearn_iter):
            # learn distribution assignments
            print("\n Searching for interventional data...")
            if not isinstance(self.mixture, IdentityMixture):
                optimizer_mix = optim.Adam(self.mixture.parameters(), lr=self.lr)
                rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                         rho=rho,
                                                         alpha=alpha,
                                                         h=h,
                                                         optimizers=[optimizer_mix],
                                                         mixture=True,
                                                         apply_threshold=self.apply_threshold)

            # relearn weights
            print("\n Adjusting weights...")
            nn.init.constant_(self.model_obs.weight, 0)
            optimizer_obs = optim.Adam(self.model_obs.parameters(), lr=self.lr)
            optimizer_int = optim.Adam(self.model_int.parameters(), lr=self.lr)

            rho, alpha, h = 1.0, 0.0, np.inf  # reset Lagrangian stuff
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_obs, optimizer_int],
                                                     mixture=True)

        W_est = self.model_obs.weight[:self.d, ...].detach().cpu().numpy().T
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

        for batch in eval_dataloader:
            features, mixture_in, targets = batch
            mixture_in = mixture_in.to(self.device)
            probs = self.mixture(mixture_in)

            assignments = torch.round(probs)
            labels_batch = torch.sum(assignments * (2 ** torch.tensor(list(range(len(variables))), device=self.device)),
                                     dim=1).squeeze().tolist()
            labels.extend(labels_batch)

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

        counts = np.array([data.targets.tolist().count(i) for i in range(np.max(data.targets) + 1)])
        p_correct = (p_correct / counts).tolist()
        if self.clustering == "observational":
            data.targets = np.zeros(len(labels))    # one cluster would be optimal

        for i, p in enumerate(p_correct):
            wandb.run.summary[f"p_{dist_keys[i]}"] = p
        wandb.run.summary["IDIOD ARI"] = sklearn.metrics.adjusted_rand_score(data.targets, labels)
        wandb.run.summary["IDIOD AMI"] = sklearn.metrics.adjusted_mutual_info_score(data.targets, labels)
        wandb.run.summary["IDIOD n_clusters"] = len(set(labels))

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
                h_new = self._h(self.model_obs.weight[:self.d, ...])
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
        #    nn.init.constant_(param, 0)

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
                #    if 'target' in self.clustering:
                #        probs = 1 - probs[..., 1:]
                #        probs = probs.to(self.device)
                #    else:
                #        probs = self.mixture(x)
                #        if self.single_target:
                #            probs = 1 - probs[..., 1:]

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
                h = self._h(self.model_obs.weight[:self.d, ...])
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

            h = self._h(self.model_obs.weight[:self.d, ...])
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


def get_likelihood(batch: torch.Tensor, preds: tuple) -> torch.Tensor:
    mean, log_var = preds
  #  dists = torch.distributions.normal.Normal(mean, std)
  #  log_l = dists.log_prob(batch)
    loss = nn.GaussianNLLLoss(reduction='none')
    return torch.exp(-loss(mean, batch, torch.exp(log_var)))


loss_dict = {'mse': nn.MSELoss(reduction='none'),
             'likelihood': get_likelihood}


class NormalIDIOD(nn.Module):
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
                 relearn_iter=1,
                 name='idiod',
                 clustering='none',
                 apply_threshold=False,
                 save_model=False,
                 log_progress=False):
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
        self.relearn_iter = relearn_iter    # alternating between learning assignments and obs / int models
        self.step = 0   # for logging
        self.save_model = save_model
        self.log_progress = log_progress

        # models
        self.loss_gaussian = loss_dict['likelihood']
        self.loss_mse = loss_dict['mse']

        self.model_obs_mean = LinearFixedParams(in_features=self.d,
                                                out_features=self.d,
                                                mask=torch.ones((self.d, self.d), dtype=torch.bool).fill_diagonal_(0),
                                                fixed_params=torch.zeros((self.d, self.d)),
                                                device=self.device)

        # independent noise, i.e. fix weight matrix and learn only biases
        self.model_obs_var = nn.Linear(in_features=self.d,
                                        out_features=self.d,
                                        device=self.device)

        self.model_int_mean = nn.Linear(in_features=self.d,
                                        out_features=self.d,
                                        device=self.device)

        self.model_int_var = nn.Linear(in_features=self.d,
                                       out_features=self.d,
                                       device=self.device)

        self.mixture = mixture_model.to(self.device)
        self.model_obs_threshold = LinearThreshold(self.model_obs_mean, self.w_threshold)

        # init params
        for param in chain(self.model_obs_mean.parameters(),
                           self.model_int_mean.parameters(),
                           self.model_obs_var.parameters(),
                           self.model_int_var.parameters()):
            nn.init.constant_(param, 0)

        # initialize variance with 1
        nn.init.constant_(self.model_obs_var.bias, 1)
        nn.init.constant_(self.model_int_var.bias, 1)

        # freeze weights
        self.model_obs_mean.bias.requires_grad = False   # for pretraining
        self.model_obs_var.weight.requires_grad = False
        self.model_int_mean.weight.requires_grad = False
        self.model_int_var.weight.requires_grad = False

        # early stopping
        self.path = os.path.join('causal_discovery', name, 'saved_models', str(uuid.uuid1()))
        os.makedirs(self.path)
        self.patience = patience

        # clustering (optional)
        self.clustering = clustering
        self.p = None

        self.apply_threshold = apply_threshold

    def predict(self, cd_input: tuple):
        variables, data = cd_input
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer_obs_mean = optim.Adam(self.model_obs_mean.parameters(), lr=self.lr)
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff

        # pretrain
        print("\n Starting pretraining...")
        rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                 rho=rho,
                                                 alpha=alpha,
                                                 h=h,
                                                 optimizers=[optimizer_obs_mean],
                                                 mixture=False)

        self.model_obs_mean.bias.requires_grad = True

        for _ in range(self.relearn_iter):
            # learn distribution assignments
            print("\n Searching for interventional data...")
            optimizer_mix = optim.Adam(self.mixture.parameters(), lr=self.lr)
            optimizer_obs_var = optim.Adam(self.model_obs_var.parameters(), lr=self.lr)
            optimizer_int_var = optim.Adam(self.model_int_var.parameters(), lr=self.lr)
            self.learn_assignments(dataloader=dataloader,
                                   optims_var=[optimizer_obs_var, optimizer_int_var],
                                   optim_mix=optimizer_mix,
                                   apply_threshold=True)

            # relearn weights
            print("\n Adjusting weights...")

            # reinit params
            for param in chain(self.model_obs_mean.parameters(),
                               self.model_int_mean.parameters()):
                nn.init.constant_(param, 0)

            optimizer_int_mean = optim.Adam(self.model_int_mean.parameters(), lr=self.lr)
            rho, alpha, h = 1.0, 0.0, np.inf  # reset Lagrangian stuff
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_obs_mean, optimizer_int_mean],
                                                     mixture=True)

        W_est = self.model_obs_mean.weight[:self.d, ...].detach().cpu().numpy().T
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

        for batch in eval_dataloader:
            features, mixture_in, targets = batch
            mixture_in = mixture_in.to(self.device)
            probs = self.mixture(mixture_in)

            assignments = torch.round(probs)
            labels_batch = torch.sum(assignments * (2 ** torch.tensor(list(range(len(variables))), device=self.device)),
                                     dim=1).squeeze().tolist()
            labels.extend(labels_batch)

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

        counts = np.array([data.targets.tolist().count(i) for i in range(np.max(data.targets) + 1)])
        p_correct = (p_correct / counts).tolist()
        if self.clustering == "observational":
            data.targets = np.zeros(len(labels))  # one cluster would be optimal

        for i, p in enumerate(p_correct):
            wandb.run.summary[f"p_{dist_keys[i]}"] = p
        wandb.run.summary["IDIOD ARI"] = sklearn.metrics.adjusted_rand_score(data.targets, labels)
        wandb.run.summary["IDIOD AMI"] = sklearn.metrics.adjusted_mutual_info_score(data.targets, labels)
        wandb.run.summary["IDIOD n_clusters"] = len(set(labels))

        return nx.relabel_nodes(pred_graph, mapping)

    def optimize_lagrangian(self, dataloader, rho, alpha, h, optimizers, mixture, apply_threshold=False):
        self.eval()
        params_init = [self.model_obs_mean.weight.detach().clone(),
                       self.model_obs_mean.bias.detach().clone(),
                       self.model_int_mean.weight.detach().clone(),
                       self.model_int_mean.bias.detach().clone()]

        for _ in range(self.max_iter):
            h_new = None
            while rho < self.rho_max:
                self.optimize(dataloader, rho, h, alpha, optimizers, mixture, params_init, apply_threshold)
                h_new = self._h(self.model_obs_mean.weight[:self.d, ...])
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break

            h = h_new.detach()
            params_init = [self.model_obs_mean.weight.detach().clone(),
                           self.model_obs_mean.bias.detach().clone(),
                           self.model_int_mean.weight.detach().clone(),
                           self.model_int_mean.bias.detach().clone()]

            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                return rho, alpha, h

    def learn_assignments(self, dataloader, optims_var, optim_mix, apply_threshold):
        train_losses = []
        best_epoch, stop_count = 0, 0

        # fix mean, learn variance
        self.model_obs_mean.weight.requires_grad = False
        self.model_obs_mean.bias.requires_grad = False
        self.model_int_mean.bias.requires_grad = False

        for epoch in range(self.max_epochs):
            self.train()

            for _, batch in enumerate(dataloader):
                for optimizer in optims_var:
                    optimizer.zero_grad()

                optim_mix.zero_grad()

                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs = self.mixture(mixture_in)
                loss = probs * loss_gaussian_obs + (1 - probs) * loss_gaussian_int
                loss = 0.5 / features.shape[0] * torch.sum(-torch.log(loss))  # equivalent to original numpy implementation
                loss.backward()

                for optimizer in optims_var:
                    optimizer.step()

            # evaluate performance on whole dataset (always without thresholding W_est) # TODO: try thresholding
            self.eval()
            loss_all = 0
            for _, batch in enumerate(dataloader):
                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(
                    features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs = self.mixture(mixture_in)
                loss_all += torch.sum(-torch.log(probs * loss_gaussian_obs + (1 - probs) * loss_gaussian_int))

            loss = 0.5 / len(dataloader.dataset) * loss_all # equivalent to original numpy implementation
            train_losses.append(loss)
            print(f"[Epoch {epoch + 1:2d}] Training loss: {loss:05.5f}")

            if len(train_losses) == 1 or loss < train_losses[best_epoch]:
                print("\t   (New best performance, saving model...)")
                torch.save(self.state_dict(), os.path.join(self.path, 'model.pt'))
                best_epoch = epoch
                stop_count = 0
            else:
                stop_count += 1

            if stop_count >= self.patience:
                break

        self.load_state_dict(torch.load(os.path.join(self.path, 'model.pt')))

        for epoch in range(self.max_epochs):
            self.train()

            for _, batch in enumerate(dataloader):
                for optimizer in optims_var:
                    optimizer.zero_grad()

                optim_mix.zero_grad()

                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs = self.mixture(mixture_in)
                loss = probs * loss_gaussian_obs + (1 - probs) * loss_gaussian_int
                loss = 0.5 / features.shape[0] * torch.sum(-torch.log(loss))  # equivalent to original numpy implementation
                loss.backward()

                for optimizer in optims_var:
                    optimizer.step()

                optim_mix.step()

            # evaluate performance on whole dataset (always without thresholding W_est) # TODO: try thresholding
            self.eval()
            loss_all = 0
            for _, batch in enumerate(dataloader):
                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(
                    features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs = self.mixture(mixture_in)
                loss_all += torch.sum(-torch.log(probs * loss_gaussian_obs + (1 - probs) * loss_gaussian_int))

            loss = 0.5 / len(dataloader.dataset) * loss_all # equivalent to original numpy implementation
            train_losses.append(loss)
            print(f"[Epoch {epoch + 1:2d}] Training loss: {loss:05.5f}")

            if len(train_losses) == 1 or loss < train_losses[best_epoch]:
                print("\t   (New best performance, saving model...)")
                torch.save(self.state_dict(), os.path.join(self.path, 'model.pt'))
                best_epoch = epoch
                stop_count = 0
            else:
                stop_count += 1

            if stop_count >= self.patience:
                break

        self.load_state_dict(torch.load(os.path.join(self.path, 'model.pt')))

        # prepare to learn mean with fixed variances
        self.model_obs_mean.weight.requires_grad = True
        self.model_obs_mean.bias.requires_grad = True
        self.model_int_mean.bias.requires_grad = True

    def optimize(self, dataloader, rho, h, alpha, optimizers, mixture, params_init, apply_threshold):
        # init params
        for param, init in zip(chain(self.model_obs_mean.parameters(), self.model_int_mean.parameters()), params_init):
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
                preds_obs = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(features)
                loss = self.loss_mse(features, preds_obs)

                if mixture:
                    preds_int = self.model_int_mean(features)
                    loss_int = self.loss_mse(features, preds_int)
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
                h = self._h(self.model_obs_mean.weight[:self.d, ...])
                obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.model_obs_mean.weight))
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
                preds_obs = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(features)
                loss = self.loss_mse(features, preds_obs)

                if mixture:
                    preds_int = self.model_int_mean(features)
                    loss_int = self.loss_mse(features, preds_int)
                    probs = self.mixture(mixture_in)
                    loss = probs * loss + (1 - probs) * loss_int

                loss_all += torch.sum(loss)

            h = self._h(self.model_obs_mean.weight[:self.d, ...])
            obj = 0.5 / len(dataloader.dataset) * loss_all + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.model_obs_mean.weight))
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




'''
class IDIOD_old(nn.Module):
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
                 path=os.path.join('causal_discovery', 'idiod', 'saved_models'),
                 clustering='none'):
        super().__init__()
        self.d = d
        self.w_est = nn.Parameter(torch.zeros(size=(self.d, self.d), device=device))
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.w_fixed = torch.zeros(size=(d, d), requires_grad=False, device=device)
        self.register_buffer('weight_update_mask', torch.ones_like(self.w_fixed, dtype=torch.bool).fill_diagonal_(0))
        self.path = os.path.join(path, str(uuid.uuid1()))
        os.makedirs(self.path)
        self.patience = patience
        self.mlp = mixture_model.to(device)
        self.sigmoid = nn.Sigmoid()
        self.bias_obs = nn.Parameter(torch.zeros(size=(self.d, ), device=device))
        self.bias_int = nn.Parameter(torch.zeros(size=(self.d, ), device=device))
        self.step = 0   # for logging assignment probabilities
        self.clustering = clustering
        self.p = None

        list(self.mlp.state_dict().values())[-1].copy_(torch.zeros_like(self.bias_obs))

    def predict(self, cd_input: tuple):
        variables, data = cd_input
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff
        # pretrain
        print("\n Starting pretraining...")
        params = [self.w_est]

        rho, alpha, h = self.optimize_lagrangian(data, self._loss, rho, alpha, h, params)
        # learn distribution assignments
        print("\n Searching for interventional data...")
        params = self.mlp.parameters()
        rho, alpha, h = self.optimize_lagrangian(data, self._idiod_loss, rho, alpha, h, params)

        self.w_est = nn.Parameter(torch.zeros(size=(self.d, self.d), device=self.device))
        # train
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff
        print("\n Adjusting weights...")
        params = [self.w_est, self.bias_obs, self.bias_int]
        rho, alpha, h = self.optimize_lagrangian(data, self._idiod_loss, rho, alpha, h, params)
        W_est = self.w_est.detach().cpu().numpy()
        W_est[np.abs(W_est) < self.w_threshold] = 0

        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))
        return nx.relabel_nodes(pred_graph, mapping)

    def optimize(self, rho, h, alpha, data, loss_fn, params):
        data.features = data.features.to(self.device)
     #   dataset = OnlyFeatures(features=data)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(params, lr=0.001)
        train_losses = []
        best_epoch, stop_count = 0, 0
        for epoch in range(self.max_epochs):
            self.train()
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()
                W = torch.where(self.weight_update_mask, self.w_est, self.w_fixed)
                if self.clustering == 'target':
                    x, self.p = batch
                else:
                    x = batch
                x = x.to(self.device)
                loss = loss_fn(x, W)
                h = self._h(W)
                obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.w_est))
                obj.backward()
                optimizer.step()
            self.eval()
            if self.clustering == 'target':
                self.p = data.memberships
            loss = loss_fn(data.features, W)
            obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * self.w_est.sum()
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
        return self.w_est

    def _loss(self, X, W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if self.loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * torch.sum(R ** 2)
        elif self.loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * torch.sum(torch.logaddexp(0, M) - X * M)
        elif self.loss_type == 'poisson':
            S = torch.exp(M)
            loss = 1.0 / X.shape[0] * torch.sum(S - X * M)
        else:
            raise ValueError('unknown loss type')
        return loss

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = torch.linalg.matrix_exp(W * W)  # (Zheng et al. 2018)
        h = torch.trace(E) - self.d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
     #   G_h = E.T * W * 2
        return h

    def _idiod_loss(self, X, W):
        if self.clustering == 'target':
            p = 1 - self.p[..., 1:]
            p = p.to(self.device)
        else:
            p = self.sigmoid(self.mlp(X))  # N x |V|
        wandb.log({'p_obs': torch.mean(p)}, step=self.step)
        wandb.log({'bias_obs': torch.mean(self.bias_obs)}, step=self.step)
        wandb.log({'bias_int': torch.mean(self.bias_int)}, step=self.step)
        self.step += 1
        loss = 0.5 / X.shape[0] * torch.sum(self._idiod_loss_obs(X, W) * p + self._idiod_loss_int(X, W) * (1 - p))
        return loss

    def _idiod_loss_obs(self, X, W):
        return (X - (X @ W + self.bias_obs[None, :])) ** 2

    def _idiod_loss_int(self, X, W):
        return (X - self.bias_int[None, :]) ** 2

    def optimize_lagrangian(self, data, loss_fn, rho, alpha, h, params):
        self.eval()
        for _ in range(self.max_iter):
            W_new, h_new = None, None
            while rho < self.rho_max:
                W_new = self.optimize(rho, h, alpha, data, loss_fn, params)
                h_new = self._h(W_new)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            W_est, h = W_new, h_new.detach()
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                return rho, alpha, h
'''