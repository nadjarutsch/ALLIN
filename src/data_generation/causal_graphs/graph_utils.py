import numpy as np
import causaldag
import networkx as nx
import random


def get_interventional_graph(graph, int_idx=-1):
    int_adj_matrix = nx.to_numpy_array(graph)
    if int_idx >= 0:
        int_adj_matrix[:, int_idx] = 0

    true_int_graph = nx.from_numpy_array(int_adj_matrix, create_using=nx.DiGraph)
    mapping = dict(zip(range(len(graph)), list(graph.nodes)))

    return nx.relabel_nodes(true_int_graph, mapping)


def get_root_nodes(graph: nx.DiGraph) -> list:
    root_nodes = []
    for n in graph:
        if graph.in_degree(n) == 0:
            root_nodes.append(n)
    return root_nodes



def add_context_vars(graph, n, vars, confounded):
    context_vars = []
    for var in random.sample(vars, n):
        graph.add_node(f'I_{var}')
        graph.add_edge(f'I_{var}', var)
        context_vars.append(f'I_{var}')

    if confounded:
        graph.add_node('conf')
        for var in context_vars:
            graph.add_edge('conf', var)

    return graph

def dag_to_mec(graph):
    adj_matrix, var_lst = causaldag.DAG.from_nx(graph).cpdag().to_amat()
    mapping = dict(zip(range(len(var_lst)), var_lst))
    mec = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.relabel_nodes(mec, mapping)


def adj_matrix_to_edges(adj_matrix):
    """
    Converts an adjacency matrix to a list of edges.
    """
    edges = np.where(adj_matrix)
    edges = np.stack([edges[0], edges[1]], axis=1)
    return edges


def edges_to_adj_matrix(edges, num_vars):
    """
    Converts a list of edges to an adjacency matrix
    """
    if not isinstance(edges, np.ndarray):
        edges = np.array(edges)
    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    if edges.shape[0] > 0:
        adj_matrix[edges[:, 0], edges[:, 1]] = True
    return adj_matrix


def edges_or_adj_matrix(edges, adj_matrix, num_vars):
    """
    Converts edges to adjacency matrix, or vice versa depending on which of the two is given and which is None.
    """
    assert edges is not None or adj_matrix is not None, 'Either the edges or adjacency matrix must be provided for the DAG'
    if edges is None:
        edges = adj_matrix_to_edges(adj_matrix)
    elif not isinstance(edges, np.ndarray):
        edges = np.array(edges)
    if adj_matrix is None:
        adj_matrix = edges_to_adj_matrix(edges, num_vars)
    return edges, adj_matrix


def sort_graph_by_vars(variables, edges=None, adj_matrix=None, latents=None):
    """
    Takes a list of variables and graph structure, and determines the causal order of the variable, 
    i.e., an order in which we can perform ancestral sampling. Returns the newly sorted graph structure.
    """
    edges, adj_matrix = edges_or_adj_matrix(edges, adj_matrix, len(variables))
    matrix_copy = np.copy(adj_matrix)

    sorted_idxs = []

    def get_empty_nodes():
        return [i for i in np.where(~matrix_copy.any(axis=0))[0] if i not in sorted_idxs]

    if latents is None or latents.shape[0] == 0 or latents[0, 0] < 0:
        empty_nodes = get_empty_nodes()
    else:
        empty_nodes = [latents[i, 0] for i in range(latents.shape[0])]
    while len(empty_nodes) > 0:
        node = empty_nodes.pop(0)
        sorted_idxs.append(node)
        matrix_copy[node, :] = False
        empty_nodes = get_empty_nodes()
    assert not matrix_copy.any(), "Sorting the graph failed because it is not a DAG!"

    variables = [variables[i] for i in sorted_idxs]
    adj_matrix = adj_matrix[sorted_idxs][:, sorted_idxs]

    num_vars = len(variables)
    edges = edges - num_vars  # To have a better replacement
    if latents is not None:
        latents = latents - num_vars
    for v_idx, n_idx in enumerate(sorted_idxs):
        edges[edges == (n_idx - num_vars)] = v_idx
        if latents is not None:
            latents[latents == (n_idx - num_vars)] = v_idx

    if latents is not None:
        latents[:, 1:] = np.sort(latents[:, 1:], axis=-1)
        return variables, edges, adj_matrix, latents, sorted_idxs
    else:
        return variables, edges, adj_matrix, sorted_idxs


def get_node_relations(adj_matrix):
    """
    Returns a matrix which describes the relations fo each node pair beyond parent-child relations.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape [num_vars, num_vars], type np.bool
                 The adjacency matrix of the graph.

    Returns
    -------
    node_relations : np.ndarray, shape [num_vars, num_vars], type np.int32
                     A matrix, where an element (i,j) can take the following values:
                       node_relations[i,j] = 1: j is an ancestor of i
                       node_relations[i,j] = -1: j is a descendant of i,
                       node_relations[i,j] = 0: j and i are independent conditioned on the empty set
                       node_relations[i,j] = 2: j and i share a confounder
    """
    # Find all ancestor-descendant relations
    ancestors = adj_matrix.T
    changed = True
    while changed: 
        new_anc = np.logical_and(ancestors[..., None], ancestors[None]).any(axis=1)
        new_anc = np.logical_or(ancestors, new_anc)
        changed = not (new_anc == ancestors).all().item()
        ancestors = new_anc

    # Output: matrix with (i,j)
    #         = 1: j is an ancestor of i
    #         = -1: j is a descendant of i,
    #         = 0: j and i are independent
    #         = 2: j and i share a confounder
    ancestors = ancestors.astype(np.int32)
    descendant = ancestors.T
    node_relations = ancestors - descendant
    confounder = (node_relations == 0) * ((ancestors[None] * ancestors[:, None]).sum(axis=-1) > 0)
    node_relations += 2 * confounder
    node_relations[np.arange(node_relations.shape[0]), np.arange(node_relations.shape[1])] = 0

    return node_relations

