import networkx as nx
import cdt
import matplotlib.pyplot as plt
import data_generation.causal_graphs.graph_visualization as visual


def causal_correctness(true_graph: nx.DiGraph,
                       pred_graph: nx.DiGraph) -> int:
    true_skeleton = true_graph.to_undirected()
    pred_skeleton = pred_graph.to_undirected()

    if nx.algorithms.similarity.graph_edit_distance(true_skeleton, pred_skeleton) == 0:

        edges = pred_graph.edges()
        for (x, y) in pred_graph.edges():
            if (y, x) not in edges:
                if (x,y) not in true_graph.edges():
                    return -cdt.metrics.SHD(true_graph, pred_graph)
        
        return true_graph.size() - cdt.metrics.SHD(true_graph, pred_graph, double_for_anticausal=False)
    
    return -cdt.metrics.SHD(true_graph, pred_graph)
    
