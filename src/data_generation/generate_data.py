import torch
import random
from omegaconf import DictConfig
    
from utils import set_seed
from data_generation.causal_graphs.variable_distributions import GaussianDist
from data_generation.causal_graphs.graph_definition import CausalDAG, CausalVariable


def generate_data(dags: list[CausalDAG], cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor, list[CausalVariable]]:
    """Samples observational and interventional data from the previously generated DAGs according to the specified
    distributions.

    Args:
        dags: list of DAGs that are used for sampling the data
        cfg: run configurations (e.g. hyperparameters) as specified in config-file and command line

    Returns:
         features: tensor of shape M x D
         target_labels: tensor of shape M x 1
         intv_variables: list of intervened variables, length R
    """

    n_obs = int(cfg.dist.n / (1 + cfg.dist.intv_ratio * cfg.n_intv_targets))
    intv_ratio = cfg.dist.intv_ratio / len(cfg.dist.obs_means)

    # sample intervened variables
    set_seed(cfg.seed)
    intv_variables = random.sample(dags[0].variables, cfg.n_intv_targets)

    features = []
    target_labels = []

    # for multimodal distributions, sample data from each mode as specified in the different DAG objects
    for dag in dags:
        # sample observational data from DAG
        set_seed(cfg.seed)
        features_obs = torch.from_numpy(dag.sample(batch_size=n_obs, as_array=True)).float()

        variables = [v.name for v in dag.variables]
        intv_indices = [variables.index(v.name) + 1 for v in intv_variables]  # 1-numb. vars.: interventional labels

        # save information about target partitions (observational, interventional)
        target_labels_mode = [0] * n_obs  # 0: observational label
        n_intv = int(n_obs * intv_ratio)
        for i in intv_indices:
            target_labels_mode.extend([i] * n_intv)

        # sample interventional data from DAG
        features_intv = []

        # imposed distribution on an intervened variable is a Gaussian with the specified mean and standard deviation
        prob_dist = GaussianDist(mu_func=lambda x: cfg.dist.intv_mean, sigma_func=lambda x: cfg.dist.intv_std)

        for v in intv_variables:  # perfect interventions on each intervened variable
            intv_dict = {v.name: prob_dist}
            intv_data = dag.sample(interventions=intv_dict,
                                   batch_size=n_intv,
                                   as_array=True)
            features_intv.append(torch.from_numpy(intv_data).float())

        features_mode = torch.cat([features_obs] + features_intv, dim=0)
        features.append(features_mode)
        target_labels.append(torch.tensor(target_labels_mode))

    return torch.cat(features, dim=0), torch.cat(target_labels, dim=0), intv_variables
