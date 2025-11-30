import numpy as np


def log_likelihood_surrogate(X, labels):
    total_log_likelihood = 0
    for cluster in np.unique(labels):
        members = X[labels == cluster]
        center = np.mean(members, axis=0)
        dists = np.sum(np.abs(members - center), axis=1)
        log_likelihood = -np.sum(dists)
        total_log_likelihood += log_likelihood
    return total_log_likelihood