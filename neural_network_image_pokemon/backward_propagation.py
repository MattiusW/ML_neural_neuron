import numpy as np
from relu_function import relu_derivative

def backward_propagation(X, Y, cache, W1, W2):
    m = X.shape[0]
    Z1, A1, A2 = cache["Z1"], cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads