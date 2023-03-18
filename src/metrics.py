import networkx as nx
import cdt
import numpy as np
import wandb

import data_generation.causal_graphs.graph_visualization as visual
import data_generation.causal_graphs.graph_definition as graphs


def log_cd_metrics(true_graph, pred_graph):
    visual.set_edge_attributes(pred_graph)

    pred_graph.remove_nodes_from(list(set(pred_graph) - set(true_graph)))
    wandb.run.summary["SHD"] = cdt.metrics.SHD(true_graph, pred_graph, double_for_anticausal=False)
    wandb.run.summary["FN"] = fn(pred_graph, true_graph)
    wandb.run.summary["FP"] = fp(pred_graph, true_graph)
    wandb.run.summary["Undirected"] = sum([not pred_graph[u][v]['directed'] for u, v in pred_graph.edges()]) / 2


def avg_neighbourhood_size(dag: graphs.CausalDAG) -> float:
    return np.sum(dag.adj_matrix) * 2 / len(dag.variables)


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
