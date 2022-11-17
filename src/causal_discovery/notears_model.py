from causal_discovery.notears.linear import *

import networkx as nx
import os
import uuid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class Notears:
    def __init__(self,
                 lambda1,
                 loss_type="l2",
                 max_iter=100,
                 h_tol=1e-8,
                 rho_max=1e+16,
                 w_threshold=0.3):

        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

    def predict(self, cd_input: tuple):

        variables, data = cd_input
        self.W_est = notears_linear(X=data,
                               lambda1=self.lambda1,
                               loss_type=self.loss_type,
                               max_iter=self.max_iter,
                               h_tol=self.h_tol,
                               rho_max=self.rho_max,
                               w_threshold=self.w_threshold)

        A_est = self.W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)


class NOTEARSTorch(nn.Module):
    def __init__(self,
                 d,
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
                 path=os.path.join('causal_discovery', 'idiod', 'saved_models')):
        super().__init__()
        self.w_est = nn.Parameter(torch.zeros(size=(d, d), device=device))
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.d = d
        self.device = device
        self.w_zeros = torch.zeros(size=(d, d), requires_grad=False, device=device)
        self.register_buffer('weight_update_mask', torch.ones_like(self.w_zeros, dtype=torch.bool).fill_diagonal_(0))
        self.path = os.path.join(path, str(uuid.uuid1()))
        os.makedirs(self.path)
        self.patience = patience

    def predict(self, cd_input: tuple):
        variables, data = cd_input
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff

        self.eval()
        W_init = self.w_est.detach().clone()
        for _ in range(self.max_iter):
            h_new = None
            while rho < self.rho_max:
                self.optimize(rho, h, alpha, data, W_init)
                h_new = self._h(self.w_est)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            W_init, h = self.w_est.detach().clone(), h_new.detach()
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                break

        W_est = self.w_est.detach().cpu().numpy()
        W_est[np.abs(W_est) < self.w_threshold] = 0
    #    print(W_est)
        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)

    def optimize(self, rho, h, alpha, data, W_init):
        self.w_est.data.copy_(W_init)
     #   nn.init.constant_(self.w_est, W_init)    # reinitialize
        data.features = data.features.to(self.device)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_losses = []
        best_epoch, stop_count = 0, 0

        for epoch in range(self.max_epochs):

            for i, x in enumerate(dataloader):
                optimizer.zero_grad()
                W = torch.where(self.weight_update_mask, self.w_est, self.w_zeros)
                x = x.to(self.device)
                loss = self._loss(x, W)
                h = self._h(W)
                obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.w_est))

                obj.backward()
                optimizer.step()

            # scheduler.step()

            self.eval()
            loss = self._loss(data.features, W)
            obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.w_est.sum()))
            train_losses.append(obj)
            print(f"[Epoch {epoch + 1:2d}] Training loss: {obj:05.5f}")

            if len(train_losses) == 1 or obj < train_losses[best_epoch]:
                print("\t   (New best performance, saving model...)")
                torch.save(self.state_dict(), os.path.join(self.path, 'model.pt'))
                best_epoch = epoch
                stop_count = 0
            else:
                stop_count += 1
        #    pbar.set_description(f"Loss: {obj_new}")
            if stop_count >= self.patience:
                break

        self.load_state_dict(torch.load(os.path.join(self.path, 'model.pt')))

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
        return h
