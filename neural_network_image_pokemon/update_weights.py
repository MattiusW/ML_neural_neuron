def update_weights(W1, b1, W2, b2, grads, learining_rate):
    W1 -= learining_rate * grads["dW1"]
    b1 -= learining_rate * grads["db1"]
    W2 -= learining_rate * grads["dW2"]
    b2 -= learining_rate * grads["db2"]

    return W1, b1, W2, b2