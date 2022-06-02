from sklearn.cluster import KMeans
import numpy as np


def kmeans(X, n_clusters, init=‘k-means++’, n_init=10):
    clustering = KMeans(n_clusters=n_clusters, init=init, n_init=n_init).fit(X)
#    labels = clustering.labels_

#    partitions = [[] for _ in range(np.max(labels) + 1)]
#    for idx, l in enumerate(labels):
#        partitions[l].append(idx)

    return clustering.labels_
