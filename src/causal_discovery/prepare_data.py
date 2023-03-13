import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain

from data_generation.datasets import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def prepare_data(cfg, features, target_labels, memberships, labels, variables: list[str]):

    # remove datapoints with label -1 (i.e. outliers as detected by the clustering)
    for tensor in [features, target_labels]:
        tensor = tensor[labels >= 0]    # TODO: doublecheck if this really changes the tensor

    # zero-center the data
    features -= torch.mean(features, dim=0, keepdim=True)

    # normalize to a standard deviation of 1
    if cfg.normalize:
        features /= torch.std(features, dim=0, keepdim=True)

    '''if cfg.causal_discovery.name == "ENCO":
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
    
            return variables, X, bk'''

    if cfg.causal_discovery.name == "asdf":
        pass
    elif cfg.causal_discovery.name == "NOTEARS":
        return variables, features.clone().numpy()
