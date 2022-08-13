from sklearn.mixture import GaussianMixture
import numpy as np
from clustering.utils import *



class GMM_soft(GaussianMixture):
    def __init__(self, n_components, random_state):
        super().__init__(n_components=n_components, random_state=random_state)

    # extended method
    def fit(self, data):
        super().fit(data)
        self.labels_ = self.predict(data)
        self.memberships_ = self.predict_proba(data)


class GMM_hard(GaussianMixture):
    def __init__(self, n_components, random_state):
        super().__init__(n_components=n_components, random_state=random_state)

    # extended method
    def fit(self, data):
        super().fit(data)
        self.labels_ = self.predict(data)
        self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], np.max(self.labels_) + 1)