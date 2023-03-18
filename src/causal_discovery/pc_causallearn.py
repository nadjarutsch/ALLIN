#from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from causal_discovery.pc.pc_causallearn import pc

import numpy as np
from typing import Union
import networkx as nx


class PC:
    def __init__(self,
                 alpha: float,
                 indep_test: str,
                 stable: bool,
                 uc_rule: int,
                 uc_priority: int,
                 verbose: bool = False,
                 show_progress: bool = True):
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.verbose = verbose
        self.show_progress = show_progress

    def predict(self,
                cd_input: tuple):

        variables, data, bk = cd_input
        int_vars = ['I_%s' % i for i in range(data.shape[1] - len(variables))]

        pred_graph = pc(data=data,
                        alpha=self.alpha,
                        indep_test=self.indep_test,
                        stable=self.stable,
                        uc_rule=self.uc_rule,
                        uc_priority=self.uc_priority,
                        background_knowledge=bk,
                        verbose=self.verbose,
                        show_progress=self.show_progress,
                        node_names=variables+int_vars)

        pred_graph.to_nx_graph()
        mapping = dict(zip(range(len(variables)), variables))

        return nx.relabel_nodes(pred_graph.nx_graph, mapping)
