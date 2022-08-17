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
        U_est, V_est = notears_linear(X = data,
                                   lambda1 = self.lambda1,
                                   loss_type = self.loss_type,
                                   num_vars = len(variables),
                                   max_iter = self.max_iter,
                                   h_tol = self.h_tol,
                                   rho_max = self.rho_max,
                                   w_threshold = self.w_threshold)

        A_est = np.zeros((data.shape[1], data.shape[1]))
        A_est[:len(variables),:len(variables)] = U_est != 0
        A_est[len(variables):,:len(variables)] = V_est < 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)