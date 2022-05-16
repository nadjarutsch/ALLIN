from sklearn.cluster import DBSCAN
import numpy as np


def dbscan(X, metric='precomputed', eps=0.5, minpts=5):
    clustering = DBSCAN(eps=eps, min_samples=minpts, metric=metric).fit(X)
    labels = clustering.labels_
    
    partitions = [[] for _ in range(np.max(labels)+1)]
    for idx, l in enumerate(labels):
        if l >= 0: # exclude data points marked as noisy (label -1)
            partitions[l].append(idx)

    return partitions