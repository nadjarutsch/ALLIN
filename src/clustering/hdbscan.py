import hdbscan
import numpy as np
from clustering.utils import *



class HDBSCAN(hdbscan.HDBSCAN):
    def __init__(self, min_cluster_size, metric, prediction_data):
        super().__init__(min_cluster_size=min_cluster_size,
                         metric=metric,
                         prediction_data=prediction_data)

    # extended method
    def fit(self, data):
        super().fit(data)
        if self.prediction_data:
            self.memberships_ = hdbscan.all_points_membership_vectors(self)
        else:
            self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], np.max(self.labels_) + 1)



