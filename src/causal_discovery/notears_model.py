from causal_discovery.notears.linear import *
from causal_discovery.notears_adv.context_notears import *
import networkx as nx
from scipy.special import expit as sigmoid

class Notears:
    def __init__(self,
                 lambda1,
                 loss_type="l2",
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
        W_est = notears_linear(X=data,
                               lambda1=self.lambda1,
                               loss_type=self.loss_type,
                               max_iter=self.max_iter,
                               h_tol=self.h_tol,
                               rho_max=self.rho_max,
                               w_threshold=self.w_threshold)

        A_est = W_est != 0
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)



class NotearsContext(Notears):
    def __init__(self,
                 lambda1,
                 loss_type="l2",
                 max_iter=100,
                 h_tol=1e-8,
                 rho_max=1e+16,
                 w_threshold=0.3,
                 v_threshold=0.3):
        super().__init__(lambda1, loss_type, max_iter, h_tol, rho_max, w_threshold)
        self.v_threshold = v_threshold

    def predict(self, cd_input: tuple):

        variables, data = cd_input
        U_est, V_est = context_notears_linear(X=data,
                                              lambda1=self.lambda1,
                                              num_vars=len(variables),
                                              max_iter=self.max_iter,
                                              h_tol=self.h_tol,
                                              rho_max=self.rho_max,
                                              w_threshold=self.w_threshold)

        C = np.eye(data.shape[1] - len(variables))
        A_est = np.zeros((data.shape[1], data.shape[1]))
        A_est[:len(variables), :len(variables)] = U_est != 0
        A_est[len(variables):, :len(variables)] = sigmoid(C @ V_est) > self.v_threshold
        pred_graph = nx.from_numpy_array(A_est, create_using=nx.DiGraph)
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph, mapping)

