import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generation.datasets import *
from tqdm import tqdm

class IDIOD(nn.Module):
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
                 device='cpu'):
        super().__init__()
        #self.w_est = nn.Parameter(torch.zeros(size=(d * (d - 1), ), device=device))
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
        self.w_fixed = torch.zeros(size=(d, d), requires_grad=False, device=device)
        self.register_buffer('weight_update_mask', torch.ones_like(self.w_fixed, dtype=torch.bool).fill_diagonal_(0))

    def predict(self, cd_input: tuple):
        variables, data = cd_input
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff
   #     rho = rho.detach()
   #     alpha = alpha.detach()

        self.eval()
        if self.loss_type == 'l2':
            data = data - torch.mean(data, axis=0, keepdims=True)

        data = data.to(self.device) # TODO: later in eval, batch-wise
        for _ in range(self.max_iter):
            W_new, h_new = None, None
            while rho < self.rho_max:
                W_new = self.optimize(rho, h, alpha, data)
                h_new = self._h(W_new)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            W_est, h = W_new, h_new.detach()
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                break

        W_est[torch.abs(W_est) < self.w_threshold] = 0
    #    print(W_est)
        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est.detach().cpu().numpy(), create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)

    #def init_params(self, d):
     #   self.w_est = torch.zeros(2 * d * d)  # double w_est into (w_pos, w_neg)

    def optimize(self, rho, h, alpha, data):
      #  self.train()
        dataset = OnlyFeatures(features=data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
      #  optimizer = optim.LBFGS(self.parameters(), lr=1)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
      #  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

      #  pbar = tqdm(range(self.max_epochs))
        patience = 0
        for _ in range(self.max_epochs):
          #  print(_)
            W = torch.where(self.weight_update_mask, self.w_est, self.w_fixed)
        #    W = torch.zeros(size=(self.d, self.d), requires_grad=False).to(self.device)
         #   bool_mat = torch.ones_like(W, requires_grad=False, device=self.device)
         #   W[torch.tril(bool_mat, diagonal=-1).to(dtype=bool)] = self.w_est[:int(self.d * (self.d - 1) / 2)]
         #   W[torch.triu(bool_mat, diagonal=1).to(dtype=bool)] = self.w_est[int(self.d * (self.d - 1) / 2):]
            loss_old = self._loss(dataset.features, W)
            obj_old = loss_old + 0.5 * rho * h * h + alpha * h + self.lambda1 * self.w_est.sum()
            self.train()

            for i, x in enumerate(dataloader):
                optimizer.zero_grad()
                W = torch.where(self.weight_update_mask, self.w_est, self.w_fixed)
          #      W = torch.zeros(size=(self.d, self.d), requires_grad=False).to(self.device)
          #      W[torch.tril(bool_mat, diagonal=-1).to(dtype=bool)] = self.w_est[:int(self.d * (self.d - 1) / 2)]
          #      W[torch.triu(bool_mat, diagonal=1).to(dtype=bool)] = self.w_est[int(self.d * (self.d - 1) / 2):]
                x = x.to(self.device)
                loss = self._loss(x, W)
                h = self._h(W)
                obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * torch.sum(torch.abs(self.w_est))

                obj.backward()
                optimizer.step()


          #  scheduler.step()

            self.eval()
            loss_new = self._loss(dataset.features, W)
            obj_new = loss_new + 0.5 * rho * h * h + alpha * h + self.lambda1 * self.w_est.sum()
        #    pbar.set_description(f"Loss: {obj_new}")
            if obj_old - obj_new < 1e-3 * obj_old:
                patience += 1
                if patience >= 5:
                    break
            else:
                patience = 0


        return W


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
