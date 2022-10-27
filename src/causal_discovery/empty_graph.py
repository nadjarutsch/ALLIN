import networkx as nx


class EmptyGraph:
    def __init__(self, num_vars: int):
        self.num_vars = num_vars

    def predict(self, cd_input: tuple):

        variables = cd_input
        pred_graph = nx.empty_graph(len(variables), create_using=nx.DiGraph())
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph.nx_graph, mapping)