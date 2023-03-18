import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from omegaconf import DictConfig

from data_generation.datasets import InterventionalDataset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def prepare_data(cfg: DictConfig,
                 features: torch.Tensor,
                 target_labels: torch.Tensor,
                 memberships: torch.Tensor,
                 labels: torch.Tensor,
                 variables: list[str]) -> tuple:
    """Filters, standardizes and transforms the data into the required shape.

    Args:
        cfg: Run configurations (e.g. hyperparameters) as specified in config-file and command line.
        features: Dataset of size M x D.
        target_labels: True assignments of datapoints to their generating SCMs (0 is observational), size M x 1.
        memberships: One-hot encoded cluster labels returned by the clustering algorithm, size M x K.
        labels: Cluster labels  returned by the clustering algorithm, size M x 1.
        variables: List of variable names.

    Returns:
        List of variable names, the prepared dataset(s), background knowledge (for PC algorithm).
    """

    # remove datapoints with negative (e.g. outliers as detected by the clustering)
    features = features[labels >= 0]
    target_labels = target_labels[labels >= 0]

    # zero-center the data
    features = features - torch.mean(features, dim=0, keepdim=True)

    # normalize to a standard deviation of 1
    if cfg.normalize:
        features = features / torch.std(features, dim=0, keepdim=True)

    if "IDIOD" in cfg.causal_discovery.name:
        mixture_in = features.clone() if cfg.clustering.name == "None" else torch.from_numpy(memberships).float()
        return variables, TensorDataset(features, mixture_in, target_labels)

    elif cfg.causal_discovery.name == "PC":
        data = np.concatenate((features.clone.numpy(), memberships.clone.numpy()), axis=1, dtype=np.double)

        if cfg.causal_discovery.background_knowledge:
            bk = BackgroundKnowledge()
            bk.add_forbidden_by_pattern(".*", "I_.*")   # context variables can not be caused by any other variables
        else:
            bk = None

        return variables, data, bk

    elif cfg.causal_discovery.name == "NOTEARS":
        return variables, features.clone().numpy()

    elif cfg.causal_discovery.name == "Faria":
        return variables, TensorDataset(features)

    elif cfg.causal_discovery.name == "ENCO":
        int_dataloaders = {}

        # create dataloaders with interventions on different variables
        for label in set(torch.unique(target_labels).tolist()) - {0}:
            dataset = TensorDataset(features[target_labels == label])
            batch_size = min(len(dataset), cfg.causal_discovery.model.batch_size)
            int_dataloaders[label - 1] = DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=False,
                                                    drop_last=False)

        obs_dataset = TensorDataset(features[target_labels == 0])
        int_dataset = InterventionalDataset(dataloaders=int_dataloaders)
        return variables, obs_dataset, int_dataset
