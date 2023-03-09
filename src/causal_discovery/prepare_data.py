import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data_generation.datasets import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def prepare_data(cfg, data: PartitionData, variables: list[str]):

    # remove datapoints with label -1 (i.e. outliers as detected by the clustering)
    data.features = data.features[data.labels >= 0]

    # zero-center the data
    data.features[..., :-1] = data.features[..., :-1] - torch.mean(data.features[..., :-1], dim=0, keepdim=True)

    # normalize to a standard deviation of 1
    if cfg.normalize:
        data.features[..., :-1] = data.features[..., :-1] / torch.std(data.features[..., :-1], dim=0, keepdim=True)

    if cfg.causal_discovery.name == "ENCO":
        int_dataloaders = {}
        for idx, partition in enumerate(data.partitions[1:]):
            var_idx = partition.targets[0] - 1
            dataset = TensorDataset(partition.features[..., :-1])
            int_dataloaders[var_idx] = DataLoader(dataset,
                                                  batch_size=cfg.causal_discovery.model.batch_size,
                                                  shuffle=True,
                                                  pin_memory=False,
                                                  drop_last=False)

        obs_dataset = TensorDataset(data.partitions[0].features[..., :-1])
        int_dataset = InterventionalDataset(dataloaders=int_dataloaders)
        return variables, obs_dataset, int_dataset

    elif "NOTEARS Pytorch" in cfg.causal_discovery.name or "IDIOD" in cfg.causal_discovery.name or "ALLIN" in cfg.causal_discovery.name:
        mixture_in = data.features[..., :-1].clone() if cfg.clustering.name == "None" or cfg.clustering.name == "Observational" else torch.from_numpy(data.memberships).float()
        return variables, OnlyFeatures(features=data.features[..., :-1], mixture_in=mixture_in, targets=data.targets)

    elif cfg.causal_discovery.name == "Faria":
        dataset = OnlyFeatures(features=data.features[..., :-1])
        OnlyFeatures.__getitem__ = OnlyFeatures.return_only_features
        return variables, dataset

    elif "PC Causallearn" in cfg.causal_discovery.name:
        features = data.features[..., :-1].clone().numpy()
        memberships = data.memberships
        X = np.concatenate((features, memberships), axis=1, dtype=np.double)

        if cfg.causal_discovery.background_knowledge:
            bk = BackgroundKnowledge()
            bk.add_forbidden_by_pattern(".*", "I_.*")
        else:
            bk = None

        return variables, X, bk

    elif cfg.causal_discovery.name == "NOTEARS":
        X = data.features[..., :-1].clone().numpy()
        return variables, X
