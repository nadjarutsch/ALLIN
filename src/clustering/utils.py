from itertools import product
import numpy as np


class TargetClusterer:
    def __init__(self, n, int_ratio, n_int_targets, num_vars, non_roots_only=False):
        self.labels_ = None
        self.int_ratio = int_ratio
        self.n_int_targets = n_int_targets
        self.n_obs = int(n / (1 + self.int_ratio * self.n_int_targets))
        self.n_int = int(self.n_obs * self.int_ratio)
        self.num_vars = num_vars
        self.non_roots_only = non_roots_only
        self.int_targets = None
        self.roots = None

    def fit(self, features):
        true_target_labels = np.zeros(shape=len(features))

        for i, t in enumerate(self.int_targets):
            if self.non_roots_only:
                if t in self.roots:
                    true_target_labels[self.n_obs + i * self.n_int:self.n_obs + (i + 1) * self.n_int] = 0
                    continue

            true_target_labels[self.n_obs + i * self.n_int:self.n_obs + (i+1) * self.n_int] = t + 1

        self.labels_ = true_target_labels
        self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], self.num_vars + 1)
        return self


class NoClusterer:
    def __init__(self):
        self.labels_ = None

    def fit(self, features):
        self.labels_ = np.zeros(shape=len(features))
        self.memberships_ = labels_to_one_hot(self.labels_[self.labels_ >= 0], np.max(self.labels_) + 1)
        return self


def labels_to_one_hot(labels: np.array, max_label: np.array) -> np.array:
    return np.eye(int(max_label))[labels.astype(int)]

