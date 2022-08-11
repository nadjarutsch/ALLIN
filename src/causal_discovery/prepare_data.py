import pandas as pd
import numpy as np

import data_generation.datasets as data




def prepare_data(cfg, data: data.PartitionData, variables: list[str]) -> pd.DataFrame:
    if cfg.causal_discovery.name == "PC":
        return prepare_for_pc(data, variables)
        
    
    
def prepare_for_pc(data: data.PartitionData, variables: list[str]) -> pd.DataFrame:
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
