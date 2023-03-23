from __future__ import annotations

import torch
import numpy as np
from sklearn.cluster import KMeans

from clustering.utils import labels_to_one_hot


class Kmeans(KMeans):
    """k-means clusterer from scikit-learn with extended fit method.

    The fit method of the scikit-learn class is extended such that one-hot encoded cluster memberships are stored in an
    instance and can be accessed as an attribute.

    Attributes:
        memberships_: One-hot encoded cluster memberships, size M x n_clusters.
        attributes of the Kmeans class by scikit-learn, e.g. labels_
    """

    def __init__(self, init: str, n_clusters: int):
        """Initializes an instance of the k-means clusterer as specified.

       Args:
           init: Method used for initializing the cluster centers.
           n_clusters: Number of clusters used to fit the k-means clusterer.
       """
        super().__init__(init=init, n_clusters=n_clusters)

        self.memberships_ = None

    # extended method
    def fit(self, data: torch.Tensor, y: None = None, sample_weight: None = None) -> Kmeans:
        """Estimates cluster means with the EM algorithm and assigns datapoints to each cluster.

        Args:
            data: Datapoints used to learn the model, size M x D.
            y: Not used, present for API consistency with scikit-learn.
            sample_weight: Not used, present for API consistency with scikit-learn.

        Returns:
             self: The fitted k-means clusterer.
        """
        super().fit(data)

        self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], np.max(self.labels_) + 1)

        return self
