from clustering.depcon_kernel import kernel_k_means
from clustering.depcon_kernel import dep_contrib_kernel
from clustering.utils import labels_to_one_hot


class DepconKmeans:
    def __init__(self, n_clusters, max_iters=100, device='cpu'):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.memberships_ = None
        self.max_iters = max_iters
        self.device = device

    def fit(self, features):
        self.labels_ = kernel_k_means(data=features,
                                      num_clus=self.n_clusters,
                                      kernel=dep_contrib_kernel,
                                      init='k-means++',
                                      max_iters=self.max_iters,
                                      device=self.device)
        self.labels_ = self.labels_.detach().cpu().numpy()
        self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], self.n_clusters)
        return self
