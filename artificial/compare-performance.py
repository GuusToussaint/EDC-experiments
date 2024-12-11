import logging
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from EDC.util.load_data import load_csv, load_arff, load_txt

def fit_on_dataset(X, Y, clf):
    splits = StratifiedKFold(n_splits=10, shuffle=True)

    data_object = {
        "auc_score": [],
        "accuracy_score": []
    }

    fold = 0
    for train_index, test_index in splits.split(X, y=Y):
        fold += 1

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        positive = np.count_nonzero(Y_test)
        baseline = max(positive / len(Y_test), 1 - positive / len(Y_test))
        if baseline == 1:
            logging.info("Baseline is 1, skipping fold")
            continue

        clf.fit(X_train, Y_train)

        Y_hat = clf.predict(X_test)
        current_accuracy_score = accuracy_score(Y_test, Y_hat)
        
        # Plot ROC curve
        try:
            y_hat = clf.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(Y_test, y_hat)
        except AttributeError:
            y_hat = Y_hat
            logging.warning("The classifier does not have a predict_proba method, classification as proba")
        auc_score = roc_auc_score(Y_test, y_hat)

        logging.info(
            f"Baseline: {baseline:.4f}, AUC: {auc_score:.4f}, Accuracy: {current_accuracy_score:.4f}"
        )

        # Adding the results to the corresponding lists
        data_object["auc_score"].append(auc_score)
        data_object["accuracy_score"].append(current_accuracy_score)

        with open(os.path.join('results', f"{dataset}-rf.json"), "wb") as f:
            pickle.dump(data_object, f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Load the dataset
    dataset_folder = os.path.join(
        "/data1",
        "toussaintg1",
        "gaussian-clusters"
    )

    datasets = os.listdir(dataset_folder)

    for dataset in datasets:
        X, Y, inputs = load_csv(os.path.join(dataset_folder, dataset))

        cls = RandomForestClassifier(n_estimators=100, random_state=42)
        fit_on_dataset(X, Y, cls)
