import argparse
import random
import logging
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from EDC import EDC
from EDC.optimisers import HillClimberOptimiser
from functools import partial
from sklearn import tree
from ellyn import ellyn
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from gplearn.genetic import SymbolicClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from EDC.util.load_data import load_csv, load_arff, load_txt

def fit_on_dataset(X, Y, clf, classifier):
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

        if classifier == "EDC":
            auc_score = clf.auc_score(X_test, Y_test)
            current_accuracy_score = clf.accuracy_score(X_test, Y_test)
        else:
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

        with open(os.path.join('results', f"{dataset.split('.')[0]}-{classifier}.json"), "wb") as f:
            pickle.dump(data_object, f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimiser_logger = logging.getLogger("edc.optimisers")
    optimiser_logger.propagate = False

    random_seed = random.randrange(1, 2**32 - 1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        type=str,
        help="The classifier to use",
        choices=["random_forest", "logistic_regression", "svm_linear", "svm_rbf", "decision_tree", "lda", "M4GP", "AMAXSC", "MLP", "EDC"],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="The number of workers to use for parallelisation",
        default=1,
    )
    args = parser.parse_args()

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
        case "M4GP":
            classifier = ellyn(
                class_m4gp=True,
                classification=True,
                prto_arch_on=True,
                selection="lexicase",
                fit_type="F1",
                verbosity=0,
                random_state=random_seed,
            )
        case "AMAXSC":
            classifier = SymbolicClassifier(
                parsimony_coefficient=0.01,
                random_state=random_seed,
            )
        case "MLP":
            classifier = MLPClassifier(
                random_state=random_seed,
            )
        case "EDC":
            pass
        case _:
            raise ValueError("Invalid classifier")


    # Load the dataset
    dataset_folder = os.path.join(
        "/data1",
        "toussaintg1",
        "gaussian-clusters"
    )

    datasets = os.listdir(dataset_folder)

    for dataset in datasets:
        X, Y, inputs = load_csv(os.path.join(dataset_folder, dataset))

        if args.classifier == "EDC":
            optimiser = HillClimberOptimiser(
                fraction_random_sample=0.8307209522886011,
                step_size=0.29811510475390257,
                beam_width=2,
                inputs=inputs,
            )
            classifier = EDC(
                building_blocks=[
                    "c_*x_",
                    "c_*x_*x_",
                    "c_*exp(c_ * x_)",
                    "c_*x_**2",
                ],
                random_seed=random_seed,
                num_workers=args.num_workers,
                beam_width=10,
                max_depth=4,
                optimisation_approach="normal",
                iterations=1000,
                search_algorithm="beam",
                num_features=len(inputs),
                optimiser=optimiser,
            )

        fit_on_dataset(X, Y, classifier, args.classifier)
