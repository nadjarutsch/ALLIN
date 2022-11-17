import networkx as nx
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generation.datasets import *
from causal_discovery.allin.mixture import IdentityMixture
import os
import uuid
import wandb
from itertools import chain
import shutil
import sklearn
from utils import set_seed
from causal_discovery.notears_model import Notears


def nll(batch: torch.Tensor, preds: tuple) -> torch.Tensor:
    mean, log_var = preds
    loss = nn.GaussianNLLLoss(reduction='none')
    return loss(mean, batch, torch.exp(log_var))


loss_dict = {'mse': nn.MSELoss(reduction='none'),
             'nll': nll}


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


class ALLIN(nn.Module):
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
                 name='allin',
                 clustering='none',
                 apply_threshold=False,
                 single_target=False,
                 save_model=False,
                 log_progress=False,
                 save_w_est=True,
                 seed=-1,
                 deterministic=False):
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
        self.save_w_est = save_w_est
        self.name = name
        self.clustering = clustering
        self.seed = seed
        self.deterministic = deterministic

        torch.autograd.set_detect_anomaly(True)

        # models
        self.loss_gaussian = loss_dict['nll']
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
        set_seed(seed)
        for param in self.mixture.parameters():
            if len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.constant_(param, 0)

        for param in chain(self.model_obs_mean.parameters(),
                           self.model_int_mean.parameters(),
                           self.model_obs_var.parameters(),
                           self.model_int_var.parameters()):
            nn.init.constant_(param, 0)

        # freeze weights
        self.model_obs_mean.bias.requires_grad = False  # for pretraining
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
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=not self.deterministic)
        optimizer_obs_mean = optim.Adam(self.model_obs_mean.parameters(), lr=self.lr)
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff

        # pretrain
        print("\n Starting pretraining...")
        #rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
        #                                         rho=rho,
        #                                         alpha=alpha,
        #                                         h=h,
        #                                         optimizers=[optimizer_obs_mean],
        #                                         mixture=False)

        notears = Notears(self.lambda1, self.loss_type, self.max_iter, self.h_tol, self.rho_max, self.w_threshold)
        notears_in = (variables, data.features.clone().numpy())
        notears.predict(notears_in)
        self.model_obs_mean.weight.data.copy_(torch.from_numpy(notears.W_est))  # TODO: make un-thresholded version available

        self.model_obs_mean.bias.requires_grad = True

        for _ in range(self.relearn_iter):
            # learn distribution assignments
            if not isinstance(self.mixture, IdentityMixture):
                # learn distribution assignments
                print("\n Searching for interventional data...")

                optimizer_obs_var = optim.Adam(self.model_obs_var.parameters(), lr=self.lr)
                optimizer_int_var = optim.Adam(self.model_int_var.parameters(), lr=self.lr)
                self.learn_assignments(dataloader=dataloader,
                                       optimizers=[optimizer_obs_var, optimizer_int_var],
                                       apply_threshold=True)

           #     print(self.model_obs_var.bias, self.model_int_var.bias)

                optimizer_mix = optim.Adam(self.mixture.parameters(), lr=self.lr)
                self.learn_assignments(dataloader=dataloader,
                                       optimizers=[optimizer_mix, optimizer_obs_var, optimizer_int_var],
                                       apply_threshold=True)

            # relearn weights
            print("\n Adjusting weights...")
            for param in chain(self.model_obs_mean.parameters(),
                               self.model_int_mean.parameters()):
                nn.init.constant_(param, 0)

            optimizer_obs_mean = optim.Adam(self.model_obs_mean.parameters(), lr=self.lr)
            optimizer_int_mean = optim.Adam(self.model_int_mean.parameters(), lr=self.lr)

            rho, alpha, h = 1.0, 0.0, np.inf  # reset Lagrangian stuff
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_obs_mean, optimizer_int_mean],
                                                     mixture=True)

        W_est = self.model_obs_mean.weight.detach().cpu().numpy().T
        if self.save_w_est:
            np.savetxt(f'{self.name}_{self.clustering}_seed_{self.seed}.txt', W_est)
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

        try:
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(data.targets, labels)
        except:
            pass

        try:
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(data.targets, labels)
        except:
            pass

        wandb.run.summary["n_clusters"] = len(set(labels))
        bias_list_obs = self.model_obs_mean.bias.detach().cpu().tolist()
        bias_list_int = self.model_int_mean.bias.detach().cpu().tolist()
        for i, (bias_obs, bias_int) in enumerate(zip(bias_list_obs, bias_list_int)):
            wandb.run.summary[f"bias_obs_{variables[i]}"] = bias_obs
            wandb.run.summary[f"bias_int_{variables[i]}"] = bias_int

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
                h_new = self._h(self.model_obs_mean.weight)
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

        return rho, alpha, h

    def learn_assignments(self, dataloader, optimizers, apply_threshold):
        train_losses = []
        best_epoch, stop_count = 0, 0

        # fix mean, learn variance
        self.model_obs_mean.weight.requires_grad = False
        self.model_obs_mean.bias.requires_grad = False
        self.model_int_mean.bias.requires_grad = False

        for epoch in range(self.max_epochs):
            self.train()

            for _, batch in enumerate(dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs = self.mixture(mixture_in)

                comb_l = probs * torch.exp(-loss_gaussian_obs) + (1 - probs) * torch.exp(-loss_gaussian_int)
                loss = 0.5 / features.shape[0] * -torch.sum(torch.log(comb_l))
                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

            # evaluate performance on whole dataset (always without thresholding W_est) # TODO: try thresholding
            self.eval()
            loss_all = 0
            ll_all = 0
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

                comb_l = probs * torch.exp(-loss_gaussian_obs) + (1 - probs) * torch.exp(-loss_gaussian_int)
                ll_all += torch.sum(torch.log(comb_l))

            loss = 0.5 / len(dataloader.dataset) * -ll_all  # equivalent to original numpy implementation
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
                        names_obs = iter(["bias_obs_" + str(i) for i in range(len(self.model_obs_mean.bias))])
                        names_int = iter(["bias_int_" + str(i) for i in range(len(self.model_int_mean.bias))])
                        bias_obs_dict = dict(zip(names_obs, self.model_obs_mean.bias.detach().cpu().tolist()))
                        bias_int_dict = dict(zip(names_int, self.model_int_mean.bias.detach().cpu().tolist()))
                        wandb.log(bias_obs_dict, step=self.step)
                        wandb.log(bias_int_dict, step=self.step)

                loss = 0.5 / features.shape[0] * torch.sum(loss)  # equivalent to original numpy implementation
                h = self._h(self.model_obs_mean.weight)
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
                # preds_obs = self.model_obs_threshold(features) if apply_threshold else self.model_obs(features)   # TODO: use thresholding?
                preds_obs = self.model_obs_mean(features)
                loss = self.loss_mse(features, preds_obs)

                if mixture:
                    preds_int = self.model_int_mean(features)
                    loss_int = self.loss_mse(features, preds_int)
                    probs = self.mixture(mixture_in)
                    loss = probs * loss + (1 - probs) * loss_int

                loss_all += torch.sum(loss)

            h = self._h(self.model_obs_mean.weight)
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






class ALLIN_double(nn.Module):
    def __init__(self,
                 d,
                 mixture1,
                 mixture2,
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
                 name='allin',
                 clustering='none',
                 apply_threshold=False,
                 single_target=False,
                 save_model=False,
                 log_progress=False,
                 save_w_est=True,
                 seed=-1,
                 deterministic=False,
                 sample=False):
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
        self.save_w_est = save_w_est
        self.name = name
        self.clustering = clustering
        self.seed = seed
        self.deterministic = deterministic
        self.sample = sample

        torch.autograd.set_detect_anomaly(True)

        # models
        self.loss_gaussian = loss_dict['nll']
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

        self.mixture1 = mixture1.to(self.device)
        self.mixture2 = mixture2.to(self.device)
        self.model_obs_threshold = LinearThreshold(self.model_obs_mean, self.w_threshold)

        # init params
        set_seed(seed)
        for param in chain(self.mixture1.parameters(), self.mixture2.parameters()):
            if len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.constant_(param, 0)

        for param in chain(self.model_obs_mean.parameters(),
                           self.model_int_mean.parameters(),
                           self.model_obs_var.parameters(),
                           self.model_int_var.parameters()):
            nn.init.constant_(param, 0)

        # freeze weights
        self.model_obs_mean.bias.requires_grad = False  # for pretraining
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
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=not self.deterministic)
        optimizer_obs_mean = optim.Adam(self.model_obs_mean.parameters(), lr=self.lr)
        rho, alpha, h = 1.0, 0.0, np.inf  # Lagrangian stuff

        # pretrain
        print("\n Starting pretraining...")

        notears = Notears(self.lambda1, self.loss_type, self.max_iter, self.h_tol, self.rho_max, self.w_threshold)
        notears_in = (variables, data.features.clone().numpy())
        notears.predict(notears_in)
        self.model_obs_mean.weight.data.copy_(torch.from_numpy(notears.W_est))  # TODO: make un-thresholded version available

        self.model_obs_mean.bias.requires_grad = True

        for _ in range(self.relearn_iter):
            # learn distribution assignments
            print("\n Searching for interventional data...")

            optimizer_obs_var = optim.Adam(self.model_obs_var.parameters(), lr=self.lr)
            optimizer_int_var = optim.Adam(self.model_int_var.parameters(), lr=self.lr)
            self.learn_assignments(dataloader=dataloader,
                                   optimizers=[optimizer_obs_var, optimizer_int_var],
                                   apply_threshold=True)

            optimizer_mix1 = optim.Adam(self.mixture1.parameters(), lr=self.lr)
            optimizer_mix2 = optim.Adam(self.mixture2.parameters(), lr=self.lr)
            self.learn_assignments(dataloader=dataloader,
                                   optimizers=[optimizer_mix1, optimizer_mix2, optimizer_obs_var, optimizer_int_var],
                                   apply_threshold=True)

            # relearn weights
            print("\n Adjusting weights...")
            for param in chain(self.model_obs_mean.parameters(),
                               self.model_int_mean.parameters()):
                nn.init.constant_(param, 0)

            optimizer_obs_mean = optim.Adam(self.model_obs_mean.parameters(), lr=self.lr)
            optimizer_int_mean = optim.Adam(self.model_int_mean.parameters(), lr=self.lr)

            rho, alpha, h = 1.0, 0.0, np.inf  # reset Lagrangian stuff
            rho, alpha, h = self.optimize_lagrangian(dataloader=dataloader,
                                                     rho=rho,
                                                     alpha=alpha,
                                                     h=h,
                                                     optimizers=[optimizer_obs_mean, optimizer_int_mean],
                                                     mixture=True)

        W_est = self.model_obs_mean.weight.detach().cpu().numpy().T
        if self.save_w_est:
            np.savetxt(f'{self.name}_{self.clustering}_seed_{self.seed}.txt', W_est)
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
            probs = torch.sqrt(self.mixture1(mixture_in) * self.mixture2(mixture_in))

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

        try:
            wandb.run.summary["ARI"] = sklearn.metrics.adjusted_rand_score(data.targets, labels)
        except:
            pass

        try:
            wandb.run.summary["AMI"] = sklearn.metrics.adjusted_mutual_info_score(data.targets, labels)
        except:
            pass

        wandb.run.summary["n_clusters"] = len(set(labels))
        bias_list_obs = self.model_obs_mean.bias.detach().cpu().tolist()
        bias_list_int = self.model_int_mean.bias.detach().cpu().tolist()
        for i, (bias_obs, bias_int) in enumerate(zip(bias_list_obs, bias_list_int)):
            wandb.run.summary[f"bias_obs_{variables[i]}"] = bias_obs
            wandb.run.summary[f"bias_int_{variables[i]}"] = bias_int

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
                h_new = self._h(self.model_obs_mean.weight)
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

        return rho, alpha, h

    def learn_assignments(self, dataloader, optimizers, apply_threshold):
        train_losses = []
        best_epoch, stop_count = 0, 0

        # fix mean, learn variance
        self.model_obs_mean.weight.requires_grad = False
        self.model_obs_mean.bias.requires_grad = False
        self.model_int_mean.bias.requires_grad = False

        for epoch in range(self.max_epochs):
            self.train()

            for _, batch in enumerate(dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs1 = self.mixture1(mixture_in)
                probs2 = self.mixture2(mixture_in)

                comb_l_1 = probs1 * torch.exp(-loss_gaussian_obs) + (1 - probs1) * torch.exp(-loss_gaussian_int)
                comb_l_2 = probs2 * torch.exp(-loss_gaussian_obs) + (1 - probs2) * torch.exp(-loss_gaussian_int)
                loss1 = 0.5 / features.shape[0] * -torch.sum(torch.log(comb_l_1))
                loss2 = 0.5 / features.shape[0] * -torch.sum(torch.log(comb_l_2))
                loss = 1 / 2 * (loss1 + loss2)
                loss.backward()

                for optimizer in optimizers:
                    optimizer.step()

            # evaluate performance on whole dataset (always without thresholding W_est) # TODO: try thresholding
            self.eval()
            loss_all = 0
            ll_all_1 = 0
            ll_all_2 = 0
            for _, batch in enumerate(dataloader):
                features, mixture_in, _ = batch
                features, mixture_in = features.to(self.device), mixture_in.to(self.device)
                preds_obs_mean = self.model_obs_threshold(features) if apply_threshold else self.model_obs_mean(
                    features)
                preds_int_mean = self.model_int_mean(features)
                preds_obs_var, preds_int_var = self.model_obs_var(features), self.model_int_var(features)

                loss_gaussian_obs = self.loss_gaussian(features, (preds_obs_mean, preds_obs_var))
                loss_gaussian_int = self.loss_gaussian(features, (preds_int_mean, preds_int_var))

                probs1 = self.mixture1(mixture_in)
                probs2 = self.mixture2(mixture_in)

                comb_l_1 = probs1 * torch.exp(-loss_gaussian_obs) + (1 - probs1) * torch.exp(-loss_gaussian_int)
                comb_l_2 = probs2 * torch.exp(-loss_gaussian_obs) + (1 - probs2) * torch.exp(-loss_gaussian_int)

                ll_all_1 += torch.sum(torch.log(comb_l_1))
                ll_all_2 += torch.sum(torch.log(comb_l_2))

            loss1 = 0.5 / len(dataloader.dataset) * -ll_all_1
            loss2 = 0.5 / len(dataloader.dataset) * -ll_all_2
            loss = 1 / 2 * (loss1 + loss2)
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
                    probs = torch.sqrt(self.mixture1(mixture_in) * self.mixture2(mixture_in))
                    if self.sample:
                        probs = torch.bernoulli(probs)
                    loss = probs * loss + (1 - probs) * loss_int

                    if self.log_progress:
                        wandb.log({'p_obs': torch.mean(probs)}, step=self.step)
                        names_obs = iter(["bias_obs_" + str(i) for i in range(len(self.model_obs_mean.bias))])
                        names_int = iter(["bias_int_" + str(i) for i in range(len(self.model_int_mean.bias))])
                        bias_obs_dict = dict(zip(names_obs, self.model_obs_mean.bias.detach().cpu().tolist()))
                        bias_int_dict = dict(zip(names_int, self.model_int_mean.bias.detach().cpu().tolist()))
                        wandb.log(bias_obs_dict, step=self.step)
                        wandb.log(bias_int_dict, step=self.step)

                loss = 0.5 / features.shape[0] * torch.sum(loss)  # equivalent to original numpy implementation
                h = self._h(self.model_obs_mean.weight)
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
                # preds_obs = self.model_obs_threshold(features) if apply_threshold else self.model_obs(features)   # TODO: use thresholding?
                preds_obs = self.model_obs_mean(features)
                loss = self.loss_mse(features, preds_obs)

                if mixture:
                    preds_int = self.model_int_mean(features)
                    loss_int = self.loss_mse(features, preds_int)
                    probs = torch.sqrt(self.mixture1(mixture_in) * self.mixture2(mixture_in))
                #    if self.sample:
                #        probs = torch.bernoulli(probs)
                    loss = probs * loss + (1 - probs) * loss_int

                loss_all += torch.sum(loss)

            h = self._h(self.model_obs_mean.weight)
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