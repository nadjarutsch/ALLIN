import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import data_generation.datasets as data



def cluster(dataset: data.PartitionData, 
            model: nn.Module, 
            loss_module: nn.Module, 
            optimizer: torch.optim.Adam,
            epochs: int, 
            fit_epochs: int,
            adj_matrix: torch.Tensor,
            stds: int = 3,
            batch_size: int = 64) -> dict:      

    partitions = [list(range(len(dataset))), []]   
    
    for _ in range(epochs):
        fit(dataset.partitions[0], model, loss_module, optimizer, fit_epochs, batch_size, adj_matrix)
        partitions = sort(dataset, partitions, model, stds, batch_size, adj_matrix)

    return partitions

def fit(dataset, model, loss_module, optimizer, fit_epochs, batch_size, adj_matrix):
    
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    t = tqdm(range(fit_epochs))
    for _ in t:
        for x, idx in iter(dataloader):
            preds = model(x, adj_matrix)
            optimizer.zero_grad()
            loss = loss_module(preds, x).mean()
            loss.backward()
            optimizer.step()
        t.set_description(f'Fitting model, loss: {loss.item()}')
        
        
@torch.no_grad()
def sort(dataset, partitions, model, stds, batch_size, adj_matrix):
    
    model.eval()
    partitions = move_datapoints(dataset, 
                                 partitions,
                                 model, 
                                 adj_matrix, 
                                 batch_size, 
                                 stds, 
                                 src_idx = 0, 
                                 dest_idx = 1, 
                                 condition_fn = obs_to_int_check)
  
    partitions = move_datapoints(dataset, 
                                 partitions,
                                 model, 
                                 adj_matrix, 
                                 batch_size, 
                                 stds, 
                                 src_idx = 1, 
                                 dest_idx = 0, 
                                 condition_fn = int_to_obs_check)
    
    return partitions


def obs_to_int_check(indices, distances, thresholds):   
    return list(set(indices[distances > thresholds].tolist()))


def int_to_obs_check(indices, distances, thresholds):
    indices = indices[distances <= thresholds].tolist()
    indices_set = set([i for i in indices if indices.count(i)==distances.shape[-1]]) # TODO: vectorize?
    return list(indices_set)


def move_datapoints(dataset, 
                    partitions,
                    model, 
                    adj_matrix, 
                    batch_size, 
                    stds, 
                    src_idx = 0, 
                    dest_idx = 1, 
                    condition_fn = obs_to_int_check):
    
    if len(dataset.partitions[src_idx]) == 0:    
        return partitions
    
    dataloader = DataLoader(dataset.partitions[src_idx], batch_size=batch_size, shuffle=True, drop_last=False)
    move_indices = []   
    
    t = tqdm(iter(dataloader))
    for features, indices in t:
        preds = model(features, adj_matrix)
        mean, std = preds[...,0], torch.sqrt(torch.exp(preds[...,1]))
        indices = indices[:,None].expand(-1, adj_matrix.shape[0])
        indices = condition_fn(indices=indices, distances=torch.abs(features-mean), thresholds=stds*std)
        move_indices.extend(indices)
        n_outliers = len(move_indices)
       # if n_outliers > 0:
       #     n_correct = len(set(move_indices).intersection(dataset.targets[dest_idx]))
       #     precision =  n_correct / n_outliers
       # else:
       #     n_correct = 0
       #     precision = 'None'

        # t.set_description('Moving %i data points from partition %i to %i... Precision: %s' % (len(move_indices), src_idx, dest_idx, precision))
        t.set_description('Moving %i data points from partition %i to %i...' % (len(move_indices), src_idx, dest_idx))


    for idx in move_indices:
        partitions[dest_idx].append(idx)
        partitions[src_idx].remove(idx)

    labels = []
    for idx in range(len(dataset.features)):
        if idx in partitions[0]:
            labels.append(0)
        else:
            labels.append(1)

    dataset.update_partitions(labels)

    return partitions





