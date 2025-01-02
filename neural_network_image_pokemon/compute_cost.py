import numpy as np

def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A2 + 1e-8)) / m
    return cost