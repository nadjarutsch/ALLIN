# https://causal.dev/code/fibroblast_clustering.py

## Copyleft 2021, Alex Markham, see https://medil.causal.dev/license.html
# Tested with versions:
# python: 3.9.5
# requests: 2.25.1
# numpy: 1.20.3
# scipy: 1.6.3
# medil: 0.6.0
# matplotlib: 3.4.2
# networkx: 2.5
import requests, os
import numpy as np
import torch
#from numpy import linalg as LA
import torch.linalg as LA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from medil.ecc_algorithms import find_clique_min_cover as find_cm
import matplotlib.pyplot as plt
import networkx as nx



def dep_contrib_kernel(X, alpha=0.1, device='cuda:0'):
    num_samps, num_feats = X.shape
    thresh = torch.eye(num_feats).to(device)
    if alpha is not None:
        thresh[thresh == 0] = (
            chi2(1).ppf(1 - alpha) / num_samps
        )  # critical value corresponding to alpha
        thresh[thresh == 1] = 0
    Z = torch.zeros((num_feats, num_samps, num_samps)).to(device)
    for j in range(num_feats):
        n = num_samps
        t = torch.tile
        D = torch.from_numpy(squareform(pdist(X[:, j].reshape(-1, 1), "cityblock"))).to(device)
        D_bar = D.mean()
        D -= (
            t(D.mean(0), (n, 1)) + t(D.mean(1), (n, 1)).T - t(D_bar, (n, n))
        )  # doubly centered
        Z[j] = D / (D_bar)  # standardized
    F = Z.reshape(num_feats * num_samps, num_samps)
    left = torch.tensordot(Z, thresh, dims=([0], [0]))
    left_right = torch.tensordot(left, Z, dims=([2, 1], [0, 1]))
    gamma = (F.T @ F) ** 2 - 2 * (left_right) + LA.norm(thresh)  # helper kernel

    diag = torch.diag(gamma)
    kappa = gamma / torch.sqrt(torch.outer(diag, diag))  # cosine similarity
    kappa[kappa > 1] = 1  # correct numerical errors
    return kappa, gamma


def kernel_k_means(data, num_clus=5, kernel=dep_contrib_kernel, init='k-means++', max_iters=100, device='cuda:0'):
    num_samps, num_feats = data.shape
    if init == 'random':
        rng = np.random.default_rng(1312)
        init = rng.choice(
            num_samps, num_clus, replace=False
        )  # choose initial clusters using Forgy method
        inner_prods, _ = kernel(data, device=device)
    elif init == 'k-means++':
        inner_prods, init = plus_plus(data, num_clus, device=device)

    inner_prods = inner_prods.to(device)
    left = torch.tile(torch.diag(inner_prods)[:, None], (1, num_clus)).to(device)
    distances = (
        left
        - 2 * inner_prods[:, init]
        + torch.tile(inner_prods[init, init], (num_samps, 1)).to(device)
    )
    # use law of cosines to get angle instead of Euc dist
    # clip corrects for numerical error, e.g. 1.0000004 instead of 1.0
    arc_distances = torch.arccos(torch.clip((1 - (distances ** 2 / 2)), -1, 1))
    labels = torch.argmin(arc_distances, dim=1)
    
    for itr in range(max_iters):
        # compute kernel distance using ||x - mu|| = k(x,x) - 2k(x,mu).mean() + k(mu,mu).mean() = left - 2*middle + right
        ip_clus = torch.tile(inner_prods, (num_clus, 1, 1)).to(device)
        m_idx = torch.empty((num_clus, num_samps, num_samps)).to(device)
        for c in range(num_clus):
            m_idx[c] = (labels == c)[None,:].repeat(num_samps,1)
            
        counts = np.fromiter(
            ((labels == label).sum() for label in range(num_clus)), int, num_clus
        )

        counts = torch.from_numpy(counts).to(device)
        ip_clus[~m_idx.bool()] = 0
        middle = ip_clus.sum(2).T / counts  # sum/ counts, because 0s through off mean
        r_idx = torch.empty((num_clus, num_samps, num_samps)).to(device)
        
        for c in range(num_clus):
            i = (labels == c)[:,None].repeat(1, num_samps)
            j = (labels == c)[None,:].repeat(num_samps, 1)
            r_idx[c] = i * j

        ip_clus[~r_idx.bool()] = 0
        right = ip_clus.sum((1, 2)) / (counts ** 2)
        distances = left - 2 * middle + right

        # law of cosines
        arc_distances = torch.arccos(torch.clip((1 - (distances ** 2 / 2)), -1, 1))
        new_labels = torch.argmin(arc_distances, dim=1)
        if (labels == new_labels).all():
            print("converged")
            break
        print("iteration {} with cluster sizes {}".format(itr, counts))
        labels = new_labels
    
#    partitions = [[] for _ in range(num_clus)]
#    for idx, l in enumerate(labels):
#        partitions[l].append(idx)
    return labels.cpu()


def plus_plus(ds, k, device='cuda:0'):
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """

    centroids = [ds[0]]
    c_indices = [0]
    distances, _ = dep_contrib_kernel(ds, device=device)

    for _ in range(1, k):
        dist_sq = torch.tensor([min([distances[idx1, idx2] for idx1 in c_indices]) for idx2 in range(ds.shape[0])])
        probs = dist_sq / torch.sum(dist_sq)
        cumulative_probs = torch.cumsum(probs, 0)
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(ds[i])
        c_indices.append(i)

    return distances, torch.tensor(c_indices).long()


