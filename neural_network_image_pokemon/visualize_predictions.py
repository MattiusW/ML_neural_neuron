import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(X, predictions, true_labels, class_names, num_samples=10):
    indices = np.random.choice(X.shape[0], num_samples, replace=False)
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx])
        plt.axis('off')
        predicted_class = class_names[predictions[idx]]
        true_class = class_names[np.argmax(true_labels[idx])]
        plt.title(f"Przewidywanie: {predicted_class}\nPrawdziwy: {true_class}", fontsize=10)
    plt.tight_layout()
    plt.show()