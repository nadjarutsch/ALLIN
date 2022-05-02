import numpy as np
import torch
import os
import json

if os.path.split(os.getcwd())[-1] != 'src':
    os.chdir('../src')
    
import utils 
import data_generation.datasets as data
import data_generation.causal_graphs.graph_definition as graphs
import data_generation.causal_graphs.graph_generation as graph_gen
import data_generation.causal_graphs.variable_distributions as dists


def generate_dag(num_vars: int,
                 edge_prob: float,
                 fns: str = 'linear gaussian',
                 **kwargs
                 ) -> graphs.CausalDAG:
    if fns == "linear gaussian":
        return graph_gen.generate_continuous_graph(num_vars=num_vars,
                                                    graph_func=graph_gen.generate_random_dag,
                                                    p=edge_prob,
                                                    **kwargs)




def generate_data(dag, n_obs, int_ratio, seed, save_to_file=False, properties={}):
    """
        
    Attributes:
        
    Returns:
        
        
    """
    # reproducibility
    utils.set_seed(seed)
    
    # sample observational data from DAG
    features = torch.from_numpy(dag.sample(batch_size=n_obs, as_array=True)).float()  
    
    # save information about target partitions (observational, interventional)
    n_int = int(n_obs * int_ratio)
    targets = []
    targets.append(list(range(n_obs)))
    targets.append(list(range(n_obs, int(n_obs + n_int * dag.num_vars))))

    # sample interventional data from DAG
    intervention_dict = {}
    prob_dist = dists.GaussianDist(mu_func = lambda x: 1.0, sigma_func = lambda x: 2.0) # TODO: variable intervention (e.g. shift)
    
    for v in dag.variables:
        intervention_dict[v.name] = prob_dist
        int_data = dag.sample(interventions=intervention_dict,
                              batch_size=n_int,
                              as_array=True)
        features = torch.cat((features, torch.from_numpy(int_data).float()), dim=0) # TODO: from list, out of loop

    # create dataset from observational and interventional data 
    synth_dataset = data.PartitionData(features=features, targets=targets)
    
    # save dataset to disk and add the dataset to overview (json file)
    if save_to_file:
        properties['n_obs'] = n_obs
        properties['n_int_per_var'] = n_int
        properties['num_vars'] = dag.num_vars
        properties['seed'] = seed
        properties['dag'] = dag.__str__()
        
        dataset_name = synth_dataset.save_to_file(directory='synthetic')
        update_json(os.path.join('..', 'data', 'synthetic_datasets.json'), 
                    dataset_name, 
                    properties)
        
    return synth_dataset


def update_json(file: str, dataset_name: str, properties: dict):
    """Updates the json file that contains information about each saved dataset.
        
    Attributes:
        file: Path to the json file.
        dataset_name: Name of the dataset file (uid)
        properties: Properties of the dataset, e.g. number of variables, graph
            structure, seed etc.
    """
    
    # load json file
    utils.startupCheck(file)
    with open(file, mode='r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    # add new dataset information to the dictionary
    data_dict[dataset_name] = {}
    for key, value in properties.items():
        data_dict[dataset_name][key] = value           
        
    # update the json file with the updated dictionary
    with open(os.path.join('..', 'data', 'synthetic_datasets.json'), 'w') as fp:
        json.dump(data_dict, fp, indent=2)