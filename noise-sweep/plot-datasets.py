import os
import numpy as np
from EDC.util.load_data import load_csv
import matplotlib.pyplot as plt

def plot_dataset(X, Y, inputs, dataset_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', edgecolor='k', s=50)
    plt.title(f'Dataset: {dataset_name}')
    plt.xlabel(inputs[0])
    plt.ylabel(inputs[1])
    plt.grid(True)
    plt.savefig(f'plots/{dataset_name}.png')

if __name__ == "__main__":
    base_path = '/data1/toussaintg1/EDC-datasets/artificial_noise_sweep'
    datasets = os.listdir(
        base_path
    )

    for dataset in datasets:
        print(dataset)

        dataset_files = os.listdir(
            os.path.join(
                base_path,
                dataset
            )
        )

        # current_dataset = dataset_files[30]
        current_dataset = dataset_files[34]
        X, Y, inputs = load_csv(
            os.path.join(
                base_path,
                dataset,
                current_dataset
            )
        )
        plot_dataset(X, Y, list(inputs.keys()), dataset)



