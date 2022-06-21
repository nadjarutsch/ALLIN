import pandas as pd
import numpy as np

import data_generation.datasets as data




def prepare_data(cd: str, data: data.PartitionData, variables: list[str]) -> pd.DataFrame:
    if cd == "pc":
        return prepare_for_pc(data, variables)
        
    
    
def prepare_for_pc(data: data.PartitionData, variables: list[str]) -> pd.DataFrame:
    cols_int = ['I_%s' % i for i in range(len(data.partitions))]
    
    dfs = []
    for partition, target in zip(data.partitions, data.intervention_targets): # if no intervention targets were explicitly set, data.intervention_targets consists of only one list with zeros
    #    df_data = partition.features[...,:-1].reshape((-1,len(variables),1)).expand((-1,len(variables),2)).clone().numpy()
    #    print(df_data[...,1].shape)
    #    df_data[...,1] = np.broadcast_to(target, (df_data.shape[0], len(data.partitions)))
    #    print(np.broadcast_to(target, (df_data.shape[0], len(data.partitions))))
    #    df_data = df_data.reshape(-1, len(variables) * len(data.partitions))
    #    df = pd.DataFrame(df_data)
    #    print(df)
    #    dfs.append(rename_df_cols(df, variables))
         df_data = partition.features[...,:-1]
         df = pd.DataFrame(df_data)
         df.columns = variables
         df = (df - df.mean()) / df.std()  # normalize
         df[cols_int] = target.expand(partition.features.shape[0], len(data.partitions))
         print(df)
         dfs.append(df)
    
    df = pd.concat(dfs)
    # drop data points without inferred intervention target
 #   drop_indices = ((df[cols_int] == 1).sum(axis=1) == 0).index[((df[cols_int] == 1).sum(axis=1) == 0)]
 #   df.drop(drop_indices)

    # re-order dataframe columns 
    cols = variables + cols_int
    df = df[cols]    
    df = df.loc[:, (df != 0).any(axis=0)] # drop context variables that are always 0
    
    return df


def rename_df_cols(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    
    for i, var in enumerate(variables):
        df = df.rename(columns={i * 2:'%s' % var})
        df = df.rename(columns={i * 2 + 1:'I_%s' % var})
        
    return df
