import os
import uuid
from typing import Union

import numpy as np

import torch
from torch.utils.data import Dataset



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
            ids = torch.arange(features.shape[0]).reshape((features.shape[0],1)) # [:,None].expand(-1,features.shape[-1])
           # self.features = torch.stack((features, ids), dim=-1)
            self.features = torch.cat((features, ids), -1)
        self.targets = targets
        if not ispartition:
            self.partitions = [PartitionData(features=self.features, 
                                             targets=self.targets, 
                                             ispartition=True)]
            
        

    def __len__(self):
        """Returns the number of datapoints."""
        return len(self.features)

    def __getitem__(self, idx: int):
        """Returns the features and id of a single datapoint."""
        return self.features[idx,:-1], self.features[idx,-1].int().item()
    
    #def update_partitions(self, partitions: list[list[int]]):
    #    """Creates new instances of PartitionData, according to the provided partitioning."""
    #    partitions_lst = []
    #    for lst in partitions:
    #        indices = torch.tensor(lst).long()
    #        partition_features = torch.index_select(self.features, 0, indices)
    #        partitions_lst.append(PartitionData(features=partition_features,
    #                                            targets=self.targets,
    #                                            ispartition=True))
    #    self.partitions = partitions_lst

    def update_partitions(self, partitions: list[int]):
        """Creates new instances of PartitionData, according to the provided partitioning."""
        partitions_lst = []
        for label in set(partitions):
            indices = torch.nonzero(torch.tensor(partitions) == label).long().squeeze()
            partition_features = torch.index_select(self.features, 0, indices)
            partitions_lst.append(PartitionData(features=partition_features,
                                                targets=self.targets,
                                                ispartition=True))
        self.partitions = partitions_lst
    
    def save_to_file(self, directory: str) -> str:
        """Saves the Dataset to a file, using a randomly generated unique identifier.
        
        Attributes:
            directory: Where to save the dataset.
        
        Returns:
            Filename of the dataset."""
            
        filename = str(uuid.uuid1())
        torch.save(self, os.path.join('..', 'data', directory, filename))
        return filename
    
    def set_true_intervention_targets(self, ground_truth: list[int]): # TODO: deprecated, was only used for prototype
        # lists of indices that belong to each cluster, 0-th cluster corresponds to observational data
        partitions = []
        num_vars = self.features.shape[-1] - 1
        partitions.append(list(set(self.partitions[0].features[...,1].flatten().tolist())))

        for i, (idx_lower, idx_upper)in enumerate(zip(ground_truth, ground_truth[1:])):
            partitions.append(
                list(set(self.partitions[1].features[...,1][(self.partitions[1].features[...,1] > idx_lower) &
                                                            (self.partitions[1].features[...,1] <= idx_upper)].flatten().tolist()))
            )
            target = np.zeros(num_vars)
            target[i] = 1
            self.intervention_targets.append(target)
    
        partitions.append(list(set(self.partitions[1].features[...,1][self.partitions[1].features[...,1] < ground_truth[0]].flatten().tolist())))
        self.intervention_targets.append(torch.ones(num_vars)) # set false positives to 1-vector
        self.update_partitions(partitions)

    def set_random_intervention_targets(self):
        if len(self.partitions) > 1:
            self.intervention_targets = []

            for i in range(len(self.partitions)):
                target = torch.zeros(len(self.partitions))
                target[i] = 1
                self.intervention_targets.append(target)