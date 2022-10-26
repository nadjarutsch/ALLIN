import pandas as pd
import numpy as np
import torch.utils.data as dt

from data_generation.datasets import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def prepare_data(cfg, data: PartitionData, variables: list[str]):
    if len(data.features) != len(data.memberships):
        data.features = data.features[data.labels >= 0]

    data.features[..., :-1] = data.features[..., :-1] - torch.mean(data.features[..., :-1], dim=0, keepdim=True)

    if cfg.normalize:
        data.features[..., :-1] = data.features[..., :-1] / torch.std(data.features[..., :-1], dim=0, keepdim=True)

    if cfg.causal_discovery.name == "PC" or cfg.causal_discovery.name == "pc_pcalg":
        return prepare_for_pc(data, variables)

    if cfg.causal_discovery.name == "enco":
        int_dataloaders = {}
        for var_idx, partition in enumerate(data.partitions[1:]):
            dataset = dt.TensorDataset(partition.features[..., :-1])
            int_dataloaders[var_idx] = dt.DataLoader(dataset,
                                                     batch_size=cfg.causal_discovery.batch_size,
                                                     shuffle=True,
                                                     pin_memory=False,
                                                     drop_last=False)

        obs_dataset = dt.TensorDataset(data.partitions[0].features[..., :-1])
        int_dataset = InterventionalDataset(dataloaders=int_dataloaders)
        return variables, obs_dataset, int_dataset

    elif "notears pytorch" in cfg.causal_discovery.name or "idiod" in cfg.causal_discovery.name:
    #    features = data.features[..., :-1] - torch.mean(data.features[..., :-1], axis=0, keepdims=True)     # zero-center
        mixture_in = data.features[..., :-1].clone() if cfg.clustering.name == "none" or cfg.clustering.name == "observational" else torch.from_numpy(data.memberships).float()
        return variables, OnlyFeatures(features=data.features[..., :-1], mixture_in=mixture_in, targets=data.targets)

    elif cfg.causal_discovery.name == "faria":
        return variables, OnlyFeatures(features=data.features[..., :-1])

    elif "pc_causallearn" in cfg.causal_discovery.name:
        features = data.features[..., :-1].clone().numpy()
        # normalize
        features = (features - np.mean(features, axis=0, keepdims=True)) / np.std(features, axis=0, keepdims=True)
        # memberships = (data.memberships - np.mean(data.memberships, axis=0, keepdims=True)) / np.std(data.memberships, axis=0, keepdims=True)
        memberships = data.memberships
        X = np.concatenate((features, memberships), axis=1, dtype=np.double)

        if cfg.causal_discovery.background_knowledge:
            bk = BackgroundKnowledge()
            bk.add_forbidden_by_pattern(".*", "I_.*")

            if cfg.causal_discovery.known_targets:
                assert cfg.clustering.name == "target", f"Known intervention targets only available for target clustering. Clustering is {cfg.clustering.name}."
                for idx, var in enumerate(variables):
                    bk.add_required_by_pattern(f"I_{idx+1}", var)

            return variables, X, bk
        else:
            return variables, X, None

    elif cfg.causal_discovery.name == "notears" or "notears context" in cfg.causal_discovery.name:
        X = np.concatenate((data.features[...,:-1].clone().numpy(), data.memberships), axis=1)
        return variables, X

    elif cfg.causal_discovery.name == "notears normed":
        features = data.features[...,:-1].clone().numpy()
        features = (features - np.mean(features, axis=0, keepdims=True)) / np.std(features, axis=0, keepdims=True)
        X = np.concatenate((features, data.memberships), axis=1)
        return variables, X

    elif cfg.causal_discovery.name == "PC known context":
        # first cluster is observational, set all context vars to 0
        data.memberships[data.memberships[:,0] == 1.0][:,0] = 0
        return prepare_for_pc(data, variables)
        

def prepare_for_pc(data: PartitionData, variables: list[str]) -> pd.DataFrame:
    data.set_random_intervention_targets()
    cols_int = ['I_%s' % i for i in range(len(data.partitions))]

    if len(data.features) != len(data.memberships):
        data.features = data.features[data.labels >= 0]
    df = pd.DataFrame(data.features[...,:-1].clone().numpy())
    df = (df - df.mean()) / df.std()  # normalize
    df.columns = variables
    df[cols_int] = data.memberships

    # re-order dataframe columns 
    cols = variables + cols_int
    df = df[cols]    
    df = df.loc[:, (df != 0).any(axis=0)] # drop context variables that are always 0
    df = df.loc[:, (df != 1).any(axis=0)] # drop context variables that are always 1
    return df


def rename_df_cols(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    
    for i, var in enumerate(variables):
        df = df.rename(columns={i * 2:'%s' % var})
        df = df.rename(columns={i * 2 + 1:'I_%s' % var})
        
    return df
