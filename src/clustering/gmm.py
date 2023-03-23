from __future__ import annotations

import torch
import numpy as np
from sklearn.mixture import GaussianMixture

from clustering.utils import labels_to_one_hot


class GMM(GaussianMixture):
    """Gaussian Mixture Model from scikit-learn with extended fit method.

    Representation of a Gaussian mixture model probability distribution. This class allows to estimate the parameters of
    a Gaussian mixture distribution. The fit method of the scikit-learn class is extended such that hard cluster labels
    and one-hot encoded cluster memberships are stored in an instance and can be accessed as attributes.

    Attributes:
        labels_: Cluster labels, size M x 1.
        memberships_: One-hot encoded cluster memberships, size M x n_clusters.
        attributes of the GaussianMixture class by scikit-learn
    """

    def __init__(self, n_clusters: int, random_state: int):
        """Initializes an instance of the GMM with the specified number of components.

        Args:
            n_clusters: Number of mixture components used to fit the Gaussian Mixture Model.
            random_state: Random seed used for model initialization.
        """
        super().__init__(n_components=n_clusters, random_state=random_state)

        self.labels_ = None
        self.memberships_ = None

    def fit(self, data: torch.Tensor, y: None = None) -> GMM:
        """Estimates model parameters with the EM algorithm and predicts cluster labels from the learned mixture model.

        Args:
            data: Datapoints used to learn the model, size M x D.
            y: Not used, present for API consistency with scikit-learn.

        Returns:
             self: The fitted mixture model.
        """

        super().fit(data, y)

        self.labels_ = self.predict(data)
        self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], np.max(self.labels_) + 1)

        return self
