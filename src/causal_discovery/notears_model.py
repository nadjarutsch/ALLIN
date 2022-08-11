from causal_discovery.notears.linear import *
import networkx as nx

class Notears():
    def __init__(self,
                 lambda1,
                 loss_type,
                 max_iter=100,
                 h_tol=1e-8,
                 rho_max=1e+16,
                 w_threshold=0.3):

        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

    def predict(self, cd_input: tuple):

        variables, data = cd_input
        W_est = notears_linear(data,
                               self.lambda1,
                               self.loss_type,
                               self.max_iter,
                               self.h_tol,
                               self.rho_max,
                               self.w_threshold)

        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)