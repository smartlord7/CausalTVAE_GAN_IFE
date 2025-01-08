import numpy as np
from matplotlib import pyplot as plt


def evaluate_model(X, vae, plot=True):
    reconstructions = vae.predict(X)
    errors = np.square(X - reconstructions)
    feature_errors = errors

    if plot:
        num_features = X.shape[1]
        for i in range(num_features):
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.hist(X[:, i], bins=30, alpha=0.5, label='Original')
            plt.title(f'Feature {i + 1} - Original')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.hist(reconstructions[:, i], bins=30, alpha=0.5, color='orange', label='Reconstructed')
            plt.title(f'Feature {i + 1} - Reconstructed')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()

            plt.suptitle(f'Feature {i + 1} - Original vs Reconstructed')
            plt.show()

    return feature_errors
