from backward_propagation import backward_propagation
from compute_cost import compute_cost
from forward_propagation import forward_propagation
from update_weights import update_weights

def train_model(X_train, y_train, X_val, y_val, W1, b1, W2, b2, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        A2, cache = forward_propagation(X=X_train, W1=W1, b1=b1, W2=W2, b2=b2)

        # cost = compute_cost(A2, y_train)

        grads = backward_propagation(X_train, y_train, cache, W1, W2)

        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, grads, learning_rate)

        A2_val, _ = forward_propagation(X=X_val, W1=W1, b1=b1, W2=W2, b2=b2)
        # val_cost = compute_cost(A2_val, y_val)

        print(f"Epoch {epoch + 1}/{epoch}")
    
    return W1, b1, W2, b2
