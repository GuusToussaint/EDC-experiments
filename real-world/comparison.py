import argparse
import logging
import random
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score
from load_data import get_data
import time
import pickle

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    random_seed = random.randrange(1, 2**32 - 1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="The random seed to use",
        default=random_seed,
    )
    parser.add_argument(
        "--classifier",
        type=str,
        help="The classifier to use",
        choices=["random_forest", "logistic_regression", "svm_linear", "svm_rbf", "decision_tree", "lda"],
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="The folder containing the dataset files",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="The folder to store the results",
        default="comparison_results",
    )
    parser.add_argument(
        "--discard_results",
        action="store_true",
        help="Whether to store the results",
        default=False,
    )
    args = parser.parse_args()
    random_seed = args.random_seed
    random.seed(random_seed)
    logging.info(f"Random seed: {random_seed}")

    X, Y, inputs, scaler = get_data(args.dataset, args.dataset_folder)
    logging.info(f"Evaluating on dataset {args.dataset}")

    match args.classifier:
        case "random_forest":
            classifier = RandomForestClassifier(random_state=random_seed)
        case "logistic_regression":
            classifier = LogisticRegression(random_state=random_seed)
        case "svm_linear":
            classifier = svm.SVC(kernel="linear", probability=True, random_state=random_seed)
        case "svm_rbf":
            classifier = svm.SVC(kernel="rbf", probability=True, random_state=random_seed)
        case "decision_tree":
            classifier = tree.DecisionTreeClassifier(random_state=random_seed)
        case "lda":
            classifier = LinearDiscriminantAnalysis()
        case _:
            raise ValueError("Invalid classifier")

    splits = KFold(n_splits=10, shuffle=True, random_state=args.random_seed)
    data_object = {
        "elapsed_time": [],
        "auc_score": [],
        "accuracy_score": []
    }
    fold = 0
    for train_index, test_index in splits.split(X):
        fold += 1
        # Generate a new random seed
        random_seed = random.randrange(1, 2**32 - 1)
        logging.info(f"Random seed for fold: {random_seed}")
        random.seed(random_seed)

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        positive = np.count_nonzero(Y_test)
        baseline = max(positive / len(Y_test), 1 - positive / len(Y_test))
        if baseline == 1:
            logging.info("Baseline is 1, skipping fold")
            continue

        start_time = time.process_time()
        classifier.fit(X_train, Y_train)
        end_time = time.process_time()
        elapsed_time = end_time - start_time

        Y_hat = classifier.predict(X_test)
        current_accuracy_score = accuracy_score(Y_test, Y_hat)
        
        # Plot ROC curve
        y_hat = classifier.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(Y_test, y_hat)

        logging.info(
            f"Baseline: {baseline:.4f}, AUC: {auc_score:.4f}, Accuracy: {current_accuracy_score:.4f}"
        )

        # Adding the results to the corresponding lists
        data_object["elapsed_time"].append(elapsed_time)
        data_object["auc_score"].append(auc_score)
        data_object["accuracy_score"].append(current_accuracy_score)

        if not args.discard_results:
            with open(os.path.join(args.output_folder, f"{args.dataset}-{args.classifier}-{args.random_seed}.json"), "wb") as f:
                pickle.dump(data_object, f)

