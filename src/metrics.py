import networkx as nx
import cdt
import numpy as np
import wandb

from data_generation.causal_graphs.graph_visualization import set_edge_attributes
from data_generation.causal_graphs.graph_definition import CausalDAG


def log_cd_metrics(pred_graph: nx.DiGraph, target_graph: nx.DiGraph):
    """Logs causal discovery performance metrics (SHD, FP, FN and number of undirected edges) to weights & biases.

    Args:
        pred_graph: Predicted causal graph.
        target_graph: True underlying causal graph.
    """
    set_edge_attributes(pred_graph)

    pred_graph.remove_nodes_from(list(set(pred_graph) - set(target_graph)))
    wandb.run.summary["SHD"] = cdt.metrics.SHD(target_graph, pred_graph, double_for_anticausal=False)
    wandb.run.summary["FN"] = fn(pred_graph, target_graph)
    wandb.run.summary["FP"] = fp(pred_graph, target_graph)
    wandb.run.summary["Undirected"] = sum([not pred_graph[u][v]['directed'] for u, v in pred_graph.edges()]) / 2


def avg_neighbourhood_size(dag: CausalDAG) -> float:
    """Returns the average neighbourhood size E[N] of a node in a given Directed Acyclic Graph (DAG).

    Args:
        dag: The Directed Acyclic Graph (DAG).
    """
    return np.sum(dag.adj_matrix) * 2 / len(dag.variables)


def fp(pred_graph: nx.DiGraph, target_graph: nx.DiGraph) -> float:
    """Returns the number of false positive (FP) edges in the predicted graph, compared to the true graph.

    If the predicted graph is a mixed graph containing undirected edges, an undirected edge should only count as a
    single false positive if the corresponding nodes are not adjacent in the true graph. Since an undirected edge is
    represented as two edges in the adjacency matrix (X -> Y, Y -> X), directly comparing the adjacency matrices of the
    predicted and true graphs counts an undirected false positive edge twice. Thresholding is used here, such that only
    one direction is counted.

    Args:
        pred_graph: Predicted causal graph.
        target_graph: True underlying causal graph.
    """
    pred = nx.to_numpy_array(pred_graph).astype(bool)
    target = nx.to_numpy_array(target_graph).astype(bool)

    fps = pred * np.invert(target)
    fps = fps + fps.transpose()     # check for undirected false positive edges
    fps[fps > 1] = 1    # only count once (thresholding)
    return np.sum(fps)/2


def fn(pred_graph: nx.DiGraph, target_graph: nx.DiGraph) -> float:
    """Returns the number of false negative (FN) edges in the predicted graph, compared to the true graph.

    Args:
        pred_graph: Predicted causal graph.
        target_graph: True underlying causal graph.
    """
    pred = nx.to_numpy_array(pred_graph).astype(bool)
    target = nx.to_numpy_array(target_graph).astype(bool)
    return np.sum(np.invert(pred) * target).item()
