import os
import uuid
from typing import Union

import numpy as np

import torch
from torch.utils.data import Dataset

from clustering.utils import *


class OnlyFeatures(Dataset):
    def __init__(self,
                 features: torch.Tensor,
                 mixture_in: np.ndarray = None,
                 targets: Union[dict[str, list[int]], list[list[int]], None] = None,
                 return_only_features: bool = False):
        self.features = features
        self.mixture_in = mixture_in
        self.targets = targets
        if return_only_features:
            self.__getitem__ = self.return_only_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx, ...], self.mixture_in[idx, ...], self.targets[idx, ...]

    def return_only_features(self, idx):
        return self.features[idx, ...]


class PartitionData(Dataset):
    """Dataset that can be partitioned into subsets. Initially, the dataset is 
    not partitioned, i.e. there is only one partition containing all data points. 
    To identify the data points throughout different partitions, each data point 
    is annotated with an individual id.   
    
    Attributes:
        features: Features of the data.
        targets: Target partitions containing the data point ids for each partition.
        ispartition: Differentiates between the whole dataset and a partition.
    """
    def __init__(self, 
                 features: torch.Tensor, 
                 targets: Union[dict[str, list[int]], list[list[int]], None] = None, 
                 ispartition: bool = False): 
        """Inits an instance of PartitionData."""  
        features = features.float()
        # one-hot encoding of intervention targets for each partition
        self.intervention_targets = [torch.tensor([0])]
        
        if ispartition:
            self.features = features 
        else: 
            ids = torch.arange(features.shape[0]).reshape((features.shape[0], 1))
            self.features = torch.cat((features, ids), -1)
        self.targets = targets
        if not ispartition:
            self.partitions = [PartitionData(features=self.features, 
                                             targets=self.targets, 
                                             ispartition=True)]
        self.memberships = None
        self.labels = None

    def __len__(self):
        """Returns the number of datapoints."""
        return len(self.features)

    def __getitem__(self, idx: int):
        """Returns the features and id of a single datapoint."""
        return self.features[idx,:-1], self.features[idx,-1].int().item()

    def update_partitions(self, partitions: np.array):
        """Creates new instances of PartitionData, according to the provided partitioning."""
        partitions_lst = []
        for label in set(partitions) - set([-1]):
            indices = torch.nonzero(torch.tensor(partitions) == label).long().squeeze()
            partition_features = torch.index_select(self.features, 0, indices)
            partitions_lst.append(PartitionData(features=partition_features,
                                                targets=self.targets[indices.numpy()],
                                                ispartition=True))
        self.partitions = partitions_lst
        self.labels = partitions


class InterventionalDataset(object):
    def __init__(self, dataloaders):

        self.data_loaders = dataloaders
        self.data_iter = {}

        for var_idx in dataloaders.keys():
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])

    def get_batch(self, var_idx):
        """
        Returns batch of interventional data for specified variable.
        """
        try:
            batch = next(self.data_iter[var_idx])
        except StopIteration:
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])
            batch = next(self.data_iter[var_idx])
        return batch[0]