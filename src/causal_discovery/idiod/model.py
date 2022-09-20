import networkx as nx
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generation.datasets import *
from tqdm import tqdm
import os
import uuid
from models.mlp import MLP
import wandb
from itertools import chain


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
                 clustering='none'):
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

        # models
        self.loss = nn.MSELoss(reduction='none')
        self.model_obs = LinearFixedParams(in_features=self.d,
                                           out_features=self.d,
                                           mask=torch.ones((self.d, self.d), dtype=torch.bool).fill_diagonal_(0),
                                           fixed_params=torch.zeros((self.d, self.d)),
                                           device=self.device)
        self.model_int = nn.Linear(in_features=d, out_features=d, device=self.device)
        self.mixture = nn.Sequential(
            mixture_model.to(self.device),
            nn.Sigmoid()
        )

        # init params
        for param in chain(self.model_obs.parameters(), self.model_int.parameters(), [self.mixture[0].layers[-1].bias]):
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
            optimizer_mix = optim.Adam(self.mixture.parameters(), lr=self.lr)
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_mix],
                                                     mixture=True)

            # relearn weights
            print("\n Adjusting weights...")
            self.model_obs.weight = nn.Parameter(torch.zeros(size=(self.d, self.d), device=self.device))
            optimizer_obs = optim.Adam(self.model_obs.parameters(), lr=self.lr)
            optimizer_int = optim.Adam(self.model_int.parameters(), lr=self.lr)

            rho, alpha, h = 1.0, 0.0, np.inf  # reset Lagrangian stuff
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_obs, optimizer_int],
                                                     mixture=True)

        W_est = self.model_obs.weight.detach().cpu().numpy().T
        W_est[np.abs(W_est) < self.w_threshold] = 0
        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)

    def optimize_lagrangian(self, dataloader, rho, alpha, h, optimizers, mixture):
        self.eval()
        params_init = [self.model_obs.weight.detach().clone(),
                       self.model_obs.bias.detach().clone(),
                       self.model_int.weight.detach().clone(),
                       self.model_int.bias.detach().clone()]
        for _ in range(self.max_iter):
            h_new = None
            while rho < self.rho_max:
                self.optimize(dataloader, rho, h, alpha, optimizers, mixture, params_init)
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

    def optimize(self, dataloader, rho, h, alpha, optimizers, mixture, params_init):
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

                if self.clustering == 'target':
                    x, probs = batch
                else:
                    x = batch
                x = x.to(self.device)
                preds_obs = self.model_obs(x)
                loss = self.loss(x, preds_obs)

                if mixture:
                    preds_int = self.model_int(x)
                    loss_int = self.loss(x, preds_int)
                    if self.clustering == 'target':
                        probs = 1 - probs[..., 1:]
                        probs = probs.to(self.device)
                    else:
                        probs = self.mixture(x)
                    loss = probs * loss + (1 - probs) * loss_int

                    wandb.log({'p_obs': torch.mean(probs)}, step=self.step)
                    names_obs = iter(["bias_obs_" + str(i) for i in range(len(self.model_obs.bias))])
                    names_int = iter(["bias_int_" + str(i) for i in range(len(self.model_int.bias))])
                    bias_obs_dict = dict(zip(names_obs, self.model_obs.bias.detach().cpu().tolist()))
                    bias_int_dict = dict(zip(names_int, self.model_int.bias.detach().cpu().tolist()))
                    wandb.log(bias_obs_dict, step=self.step)
                    wandb.log(bias_int_dict, step=self.step)

                 #   wandb.log({'bias_obs': torch.mean(self.model_obs.bias)}, step=self.step)
                 #   wandb.log({'bias_int': torch.mean(self.model_int.bias)}, step=self.step)


                loss = 0.5 / x.shape[0] * torch.sum(loss)  # equivalent to original numpy implementation
                h = self._h(self.model_obs.weight)
                obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.model_obs.weight))
                obj.backward()

                wandb.log({'loss (unconstrained)': loss}, step=self.step)
                wandb.log({'loss (Lagrange)': obj}, step=self.step)
                self.step += 1

                for optimizer in optimizers:
                    optimizer.step()

            # evaluate performance on whole dataset
            self.eval()
            loss_all = 0
            for _, batch in enumerate(dataloader):
                if self.clustering == 'target':
                    x, probs = batch
                else:
                    x = batch
                x = x.to(self.device)
                preds_obs = self.model_obs(x)
                loss = self.loss(x, preds_obs)

                if mixture:
                    preds_int = self.model_int(x)
                    loss_int = self.loss(x, preds_int)
                    if self.clustering == 'target':
                        probs = 1 - probs[..., 1:]
                        probs = probs.to(self.device)
                    else:
                        probs = self.mixture(x)
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