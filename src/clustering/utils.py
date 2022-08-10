import torch
from itertools import product
import numpy as np



class RandomClusterer():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels_ = None
    def fit(self, features):
        self.labels_ = np.random.randint(low=0, high=self.n_clusters, size=len(features))
        return self


class TargetClusterer():
    def __init__(self, n_obs, int_ratio, num_vars):
        self.labels_ = None
        self.n_obs = n_obs
        self.n_int = int(n_obs * int_ratio)
        self.num_vars = num_vars
    def fit(self, features):
        true_target_labels = np.zeros(shape=len(features))
       # for i in range(1, self.num_vars + 1):
         #   true_target_labels[i * self.n_int, (i+1) * self.n_int]
         #   true_target_labels.extend([i] * self.n_int)

        for i in range(self.num_vars):
            true_target_labels[self.n_obs + i * self.n_int:self.n_obs + (i+1) * self.n_int] = i+1

        self.labels_ = true_target_labels
        return self


class ObservationalClusterer():
    def __init__(self, n_obs):
        self.labels_ = None
        self.n_obs = n_obs
    def fit(self, features):
        #self.labels_ = [0] * self.n_obs + [-1] * (len(features) - self.n_obs)
        labels = np.zeros(shape=len(features))
        labels[self.n_obs:len(features)] = -1
        self.labels_ = labels
        return self


class NoClusterer():
    def __init__(self):
        self.labels_ = None
    def fit(self, features):
        self.labels_ = np.zeros(shape=len(features))
        return self


def match_clusters(pred, target):
    counts = []
    int_targets = []
    for pred_cluster, target_cluster in product(pred.partitions, target.partitions):
        # compare equal elements
        count = len(set(pred_cluster.features[..., -1].tolist()) & set(target_cluster.features[..., -1].tolist()))
        counts.append(count)
        if len(counts) == len(target.partitions):
            int_targets.append(list(set(pred.targets))[np.argmax(counts)])
            counts = []

    return int_targets


def labels_to_one_hot(labels: np.array, max_label: np.array) -> np.array:
    return np.eye(int(max_label))[labels.astype(int)]