import os
from EDC.util.load_data import load_csv, load_arff, load_txt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

def get_data(dataset, data_folder):
    """Loads the data of a pre-defined list of datasets

    Args:
        dataset (str): The identifier of the dataset to load

    Raises:
        ValueError: The dataset is not in the list of pre-defined datasets

    Returns:
        tuple: X, Y, inputs
    """
    match dataset:
        case "AD01":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD01.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD02":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD02.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD03":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD03.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD04":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD04.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD05":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD05.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD06":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD06.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD07":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD07.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "AD10":
            X, Y, inputs = load_csv(os.path.join(data_folder, "artificial_custom", "AD10.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "ADULT":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "adult.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "IONOSPHERE":
            X, Y, inputs = load_arff(
                os.path.join(data_folder, "real-world", "ionosphere.arff")
            )
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "CREDIT":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "credit.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "WISCONSIN":
            X, Y, inputs = load_arff(
                os.path.join(data_folder, "real-world", "wisconsin.arff")
            )
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "DIABETES":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "diabetes.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "SONAR":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "sonar.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "BANKNOTE":
            X, Y, inputs = load_txt(os.path.join(data_folder, "real-world", "data_banknote_authentication.txt"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "OCCUPANCY":
            X, Y, inputs = load_csv(os.path.join(data_folder, "real-world", "occupancy_detection.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "HEPATITIS":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "hepatitis.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "CYLINDER":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "cylinder-bands.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "BREAST":
            X, Y, inputs = load_arff(os.path.join(data_folder, "real-world", "breast.cancer.arff"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "KORNS_T1":
            X, Y, inputs = load_csv(os.path.join(data_folder, "korns", "t1.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "KORNS_T2":
            X, Y, inputs = load_csv(os.path.join(data_folder, "korns", "t2.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case "KORNS_T3":
            X, Y, inputs = load_csv(os.path.join(data_folder, "korns", "t3.csv"))
            Y = Y.astype(int)
            X, scaler = normalise_data(X)
        case _:
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