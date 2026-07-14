import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import arff
import matplotlib.pyplot as plt


def load_csv(filename):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1]
    inputs = {}
    for i, f in enumerate(X[0, :]):
        inputs[f"x{i}"] = (min(X[:, i]), max(X[:, i]))
    return X, Y, inputs

def load_arff(dataset_file) -> tuple[np.ndarray, np.ndarray]:
    """
    This function will be used to load the data from the arff file.
    """

    dataset = arff.load(open(dataset_file, "r"))
    true_class = dataset['attributes'][-1][1][0]

    # Extract the categorical features
    attributes_without_target = dataset['attributes'][:-1]
    categorical_indexes = []
    attribute_names = []
    category_names = []
    for attribute_index, attribute in enumerate(attributes_without_target):
        if type(attribute[1]) == list:
            categorical_indexes.append(attribute_index)
            category_names.append(attribute[1])
        else:
            attribute_names.append(attribute[0])

    data = np.array(dataset['data'])
    Y = data[:, -1] == true_class

    if len(categorical_indexes) == 0:
        X = data[:, :-1].astype(float)
        inputs = {}
        for i, f in enumerate(attribute_names):
            inputs[f"x{i}"] = (min(X[:, i]), max(X[:, i]))
        return X, Y, inputs

    # add one hot encoding for categorical features
    encoder = OneHotEncoder(sparse_output=False, categories=category_names, handle_unknown="ignore", min_frequency=0.02)
    categorical_features = data[:, categorical_indexes]  # use the list of categorical feature indexes
    encoder = encoder.fit(categorical_features)
    one_hot_encoded_features = encoder.transform(categorical_features)

    # remove the categorical columns from the original data
    data = np.delete(data, categorical_indexes, axis=1)

    # concatenate the one-hot encoded features with the rest of the data
    X = np.concatenate([one_hot_encoded_features, data[:, :-1].astype(float)], axis=1)
    features = np.concatenate([encoder.get_feature_names_out(), attribute_names])


    Y = Y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]

    inputs = {}
    for i, f in enumerate(features):
        inputs[f"x{i}"] = (min(X[:, i]), max(X[:, i]))
    return X, Y, inputs 

def load_txt(dataset_file) -> tuple[np.ndarray, np.ndarray]:
    """
    This function will be used to load the data from the txt file.
    """
    data = np.loadtxt(dataset_file, delimiter=",")
    Y = data[:, -1]
    X = data[:, :-1]
    
    inputs = {}
    for i, f in enumerate(X[0, :]):
        inputs[f"x{i}"] = (min(X[:, i]), max(X[:, i]))
    return X, Y, inputs

def get_data(dataset, data_folder):
    """Loads the data of a pre-defined list of datasets

    Args:
        dataset (str): The identifier of the dataset to load

    Raises:
        ValueError: The dataset is not in the list of pre-defined datasets

    Returns:
        tuple: X, Y, inputs
    """
    if dataset == "AD01":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD01.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD02":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD02.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD03":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD03.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD04":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD04.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD05":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD05.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD06":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD06.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD07":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD07.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "AD10":
        X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD10.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "ADULT":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "adult.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "IONOSPHERE":
        X, Y, inputs = load_arff(
            os.path.join(data_folder, "real-world", "ionosphere.arff")
        )
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "CREDIT":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "credit.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "WISCONSIN":
        X, Y, inputs = load_arff(
            os.path.join(data_folder, "real-world", "wisconsin.arff")
        )
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "DIABETES":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "diabetes.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "SONAR":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "sonar.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "BANKNOTE":
        X, Y, inputs = load_txt(os.path.join(data_folder, "real-world", "data_banknote_authentication.txt"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "OCCUPANCY":
        X, Y, inputs = load_csv(os.path.join(data_folder, "real-world", "occupancy_detection.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "HEPATITIS":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "hepatitis.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "CYLINDER":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "cylinder-bands.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "BREAST":
        X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "breast.cancer.arff"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "KORNS_T1":
        X, Y, inputs = load_csv(os.path.join(data_folder, "korns", "t1.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "KORNS_T2":
        X, Y, inputs = load_csv(os.path.join(data_folder, "korns", "t2.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    elif dataset == "KORNS_T3":
        X, Y, inputs = load_csv(os.path.join(data_folder, "korns", "t3.csv"))
        Y = Y.astype(int)
        X, scaler = normalise_data(X)
    else:
        raise ValueError("Invalid dataset")
    return X, Y, inputs, scaler


def normalise_data(X):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    return scaler.transform(X), scaler


def plot_densities(X):
    df = pd.DataFrame(X)
    df.plot(
        kind="density", subplots=True, layout=(11, 4), sharex=False, figsize=(20, 14)
    )
    plt.savefig("density.png")

if __name__ == "__main__":
    X, Y, inputs, scaler = get_data("KORNS-T1", "/data1/toussaintg1/EDC-datasets")

    instances = len(X)
    features = len(inputs)
    print(f"Instances: {instances}, Features: {features}")