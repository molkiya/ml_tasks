import numpy as np


def create_windows(x, p):
    X, Y = [], []
    for i in range(len(x) - p):
        X.append(x[i:i + p])
        Y.append(x[i + p])
    return np.array(X), np.array(Y)
