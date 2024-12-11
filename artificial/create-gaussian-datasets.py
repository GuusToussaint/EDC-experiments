# In this file the 100 datasets for the gaussian artifical experiments are created
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os

data_folder = os.path.join(
    "/data1",
    "toussaintg1",
    "gaussian-clusters"
)

# Generate datasets based on distributions
def generate_distribution_dataset(n_samples, n_features):
    mean_range = (-10, 10)
    scale_range = (1, 3)
    true_clusters = 2
    false_clusters = 4
    # Three true distributions
    samples_per_distribution = n_samples // (true_clusters + false_clusters)

    X = np.empty((0, n_features))
    for n_true in range(true_clusters):
        X_true = np.ones((samples_per_distribution, n_features))
        X_true[:, 0] = np.random.normal(
            loc=np.random.randint(*mean_range),
            scale=np.random.randint(*scale_range),
            size=(samples_per_distribution),
        )
        X_true[:, 1] = np.random.normal(
            loc=np.random.randint(*mean_range),
            scale=np.random.randint(*scale_range),
            size=(samples_per_distribution),
        )
        X = np.concatenate([X, X_true])

    for n_false in range(false_clusters):
        X_false = np.ones((samples_per_distribution, n_features))
        X_false[:, 0] = np.random.normal(
            loc=np.random.randint(*mean_range),
            scale=np.random.randint(*scale_range),
            size=(samples_per_distribution),
        )
        X_false[:, 1] = np.random.normal(
            loc=np.random.randint(*mean_range),
            scale=np.random.randint(*scale_range),
            size=(samples_per_distribution),
        )
        X = np.concatenate([X, X_false])

    Y = np.concatenate(
        [
            np.ones(true_clusters * samples_per_distribution),
            np.zeros(false_clusters * samples_per_distribution),
        ]
    )
    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8)
    plt.grid()
    plt.savefig("distribution_dataset.png")

    return X, Y

if __name__ == "__main__":

    for i in range(100):
        X, Y = generate_distribution_dataset(2000, 2)

        # Store the created dataset
        with open(os.path.join(data_folder, f"{i}.csv"), "w") as f:
            for i in range(len(X)):
                f.write(f"{X[i][0]},{X[i][1]},{Y[i]}\n")