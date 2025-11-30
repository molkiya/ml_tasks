import numpy as np


# Реализация манхэттенских расстояний
def manhattan_distance(X):
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))
    # Манхэттенская метрика схожести
    # ∑ | 𝑥𝑖− 𝑦𝑖 |
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.sum(np.abs(X[i] - X[j]))
    return distances
