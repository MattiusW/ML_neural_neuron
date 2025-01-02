import numpy as np


class Trainer:
    def __init__(self, model, learning_rate=0.01, epochs=10):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    @staticmethod
    def compute_cost(A2, Y):
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A2 + 1e-8)) / m
        return cost
        
    def train(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            A2, cache = self.model.forward_propagation(X_train)
            cost = self.compute_cost(A2, y_train)

            grads = self.model.backward_propagation(X_train, y_train, cache)
            self.model.update_weights(grads, self.learning_rate)

            A2_val, _ = self.model.forward_propagation(X_val)
            val_cost = self.compute_cost(A2_val, y_val)

            if (epoch % 100 == 0):
                print(f"Epoka {epoch + 1}/{self.epochs}, Błąd modelu: {cost:.4f}, Przeuczenie: {val_cost:.4f}")

