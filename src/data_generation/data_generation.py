import numpy as np
import torch
import os

if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')
    
import utils 
import data_generation.datasets as data
import data_generation.causal_graphs.variable_distributions as dists


def generate_data(dag, n_obs, int_ratio, seed, int_mu, int_sigma, int_variables):
    """
        
    Attributes:
        
    Returns:
        
        
    """
    
    # sample observational data from DAG
    utils.set_seed(seed)
    features = torch.from_numpy(dag.sample(batch_size=n_obs, as_array=True)).float()

    variables = [v.name for v in dag.variables]
    int_indices = [variables.index(v.name) + 1 for v in int_variables]

    # save information about target partitions (observational, interventional)
    true_target_labels = [0] * n_obs
    n_int = int(n_obs * int_ratio)
    for i in int_indices:
        true_target_labels.extend([i] * n_int)

    # sample interventional data from DAG
    interventions = []
    if n_int > 0:
        prob_dist = dists.GaussianDist(mu_func=lambda x: int_mu, sigma_func=lambda x: int_sigma)
        for v in int_variables:     # perfect interventions on each variable
            intervention_dict = {}
            intervention_dict[v.name] = prob_dist
            int_data = dag.sample(interventions=intervention_dict,
                                  batch_size=n_int,
                                  as_array=True)
            features = torch.cat((features, torch.from_numpy(int_data).float()), dim=0) # TODO: from list, outside of loop
            interventions.append(intervention_dict)
        
    # create dataset from observational and interventional data 
    synth_dataset = data.PartitionData(features=features, targets=np.array(true_target_labels))
        
    return synth_dataset, interventions
