from matplotlib import pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def predict(self, X):
        A2, _ = self.model.forward_propagation(X)
        return np.argmax(A2, axis=1)
    
    def calculate_accuracy(self, predictions, true_labels):
        true_classes = np.argmax(true_labels, axis=1)
        accuracy = np.mean(predictions == true_classes)
        return accuracy
    
    def visualize_predictions(self, X, predictions, true_labels, num_samples=10):
        indices = np.random.choice(X.shape[0], num_samples, replace=False)
        plt.figure(figsize=(15,10))
        
        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i + 1)
            plt.imshow(X[idx].reshape(128, 128, 3))
            plt.axis('off')
            predicted_class = self.class_names[predictions[idx]]
            true_class = self.class_names[np.argmax(true_labels[idx])]
            plt.title(f"Przewidywania: {predicted_class}\n Pokemon: {true_class}", fontsize=10)
        
        plt.tight_layout()
        plt.show()