from sklearn.cluster import KMeans
import numpy as np
from clustering.utils import *



class Kmeans(KMeans):
    def __init__(self, init, n_clusters, soft=False):
        super().__init__(init=init, n_clusters=n_clusters)
        self.soft = soft

    # extended method
    def fit(self, data):
        super(Kmeans, self).fit(data)
        if self.soft:
            pass
        else:
            self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], np.max(self.labels_) + 1)
