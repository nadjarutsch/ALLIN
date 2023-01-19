import networkx as nx
import cdt
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import wandb

import causaldag

import data_generation.causal_graphs.graph_visualization as visual
import data_generation.datasets as data
import data_generation.causal_graphs.graph_definition as graphs


def log_cd_metrics(true_graph, pred_graph, mec, title):
    pred_graph.remove_nodes_from(list(set(pred_graph) - set(true_graph)))
    wandb.run.summary["SHD"] = cdt.metrics.SHD(true_graph, pred_graph, double_for_anticausal=False)
    wandb.run.summary["CC"] = causal_correctness(true_graph, pred_graph, mec)
    wandb.run.summary["FN"] = fn(pred_graph, true_graph)
    wandb.run.summary["FP"] = fp(pred_graph, true_graph)
    wandb.run.summary["Undirected"] = sum([not pred_graph[u][v]['directed'] for u, v in pred_graph.edges()]) / 2


def causal_correctness(true_graph: nx.DiGraph,
                       pred_graph: nx.DiGraph,
                       mec: nx.DiGraph = None) -> int:
    if mec is None:
        adj_matrix, var_lst = causaldag.DAG.from_nx(true_graph).cpdag().to_amat()
        mapping = dict(zip(range(len(var_lst)), var_lst))
        mec = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        mec = nx.relabel_nodes(mec, mapping)

    true_edges = set(true_graph.edges())
    pred_edges = set(pred_graph.edges())
    mec_edges = set(mec.edges())
    shd = cdt.metrics.SHD(true_graph, pred_graph, double_for_anticausal=False)
    n_undirected = sum([mec[u][v]['directed'] == False for u, v in mec.edges()]) / 2.

    if true_edges.issubset(pred_edges) and pred_edges.issubset(mec_edges):
        return n_undirected - shd
    
    return -shd
    

def joint_log_prob(dataset: data.PartitionData,
                   dag: graphs.CausalDAG,
                   interventions: list[dict],
                   title: str,
                   log: bool = True) -> None:

    partitions = [part.features[...,0,1].clone().detach().int().tolist() for part in dataset.partitions]
    ground_truth = partitions == dataset.targets
    
    true_part = [f"int {v.name}" for v in dag.variables]
    true_part.insert(0, "obs")
    
    for p_i, part in enumerate(dataset.partitions):
        plt.figure(figsize=(6,6))
        dataloader = DataLoader(part, batch_size=1, shuffle=False, drop_last=False) # TODO: test larger batch size
        log_probs = []
        X = []
        for inter in interventions:
            X.append(f" int {list(inter.keys())}" if list(inter.keys()) != [] else "obs")
            log_prob = 0
            for features, indices in dataloader:
             #   log_prob += dag.get_joint_log_prob(features.squeeze(), inter)
                log_prob += dag.get_joint_prob(features.squeeze(), inter)
            
            log_prob = log_prob / len(part) if len(part) != 0 else 0          
            log_probs.append(log_prob)
        
        X_axis = np.arange(len(X))

        plt.bar(X_axis, log_probs, 0.4)          
        plt.xticks(X_axis, X)
        plt.xlabel("Data distribution")
        plt.ylabel("Avg sample log-probability")
        plt.title(title)
                
        partition_name = f"partition {true_part[p_i]}" if ground_truth else f"partition {p_i}"
        
        if log:
            wandb.log({partition_name: wandb.Image(plt)})
            plt.close()
        else:
            plt.show()


def avg_neighbourhood_size(dag: graphs.CausalDAG) -> float:
    return np.sum(dag.adj_matrix) * 2 / len(dag.variables)


#def flips()


def fp(pred_dag, target_dag) -> float:
    pred = nx.to_numpy_array(pred_dag).astype(bool)
    target = nx.to_numpy_array(target_dag).astype(bool)

    fps = pred * np.invert(target)
    fps = fps + fps.transpose()
    fps[fps > 1] = 1
    return np.sum(fps)/2


def fn(pred_dag, target_dag) -> float:
    pred = nx.to_numpy_array(pred_dag).astype(bool)
    target = nx.to_numpy_array(target_dag).astype(bool)
    return np.sum(np.invert(pred) * target)
