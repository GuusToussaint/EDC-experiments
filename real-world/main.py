import argparse
import logging
from EDC import EDC
from EDC.optimisers import (
    RandomSampleOptimiser,
    GradientDecentOptimiser,
    HillClimberOptimiser,
)
from EDC.optimisers.hill_climber_gradient_decent import (
    HillClimberGradientDecentOptimiser,
)
from EDC.util.plot_decision_boundary import plot as plot_decision_boundary
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from functools import partial
import random
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from load_data import get_data
import time
import pickle
import sympy
import os


def get_optimiser(optimiser_name):
    match optimiser_name:
        case "gradient_decent":
            optimiser = partial(
                GradientDecentOptimiser, 
                learning_rate=1.7610811559260424,
            )
        case "random_sample":
            optimiser = partial(RandomSampleOptimiser)
        case "hill_climber":
            optimiser = partial(
                HillClimberOptimiser, 
                fraction_random_sample=0.8307209522886011,
                step_size=0.29811510475390257,
                beam_width=2,
            )
        case "hill_climber_gradient_decent":
            optimiser = partial(
                HillClimberGradientDecentOptimiser,
                learning_rate=1.7610811559260424,
                fraction_random_sample=0.8307209522886011,
                step_size=0.29811510475390257,
                beam_width=2,
            )
        case _:
            raise ValueError("Invalid optimiser")
    return optimiser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimiser_logger = logging.getLogger("edc.optimisers")
    optimiser_logger.propagate = False
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.propagate = False
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
        "--optimiser",
        type=str,
        help="The optimiser to use",
        default="random_sample",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="The search algorithm to use",
        default="beam",
        choices=["exhaustive", "beam"],
    )
    parser.add_argument(
        "--plot_decision_boundary",
        action="store_true",
        help="Whether to plot the decision boundary",
        default=False,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Which fold to evaluate on, if not provided, evaluate on all folds",
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
        default="results",
    )
    parser.add_argument(
        "--discard_results",
        action="store_true",
        help="Whether to store the results",
        default=False,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="The number of workers to use for parallelisation",
        default=1,
    )
    args = parser.parse_args()
    random_seed = args.random_seed
    random.seed(random_seed)
    logging.info(f"Random seed: {random_seed}")

    X, Y, inputs, scaler = get_data(args.dataset, args.dataset_folder)
    logging.info(f"Evaluating on dataset {args.dataset}")
    logging.info(f"Optimiser: {args.optimiser}")

    optimiser = get_optimiser(args.optimiser)
    optimiser = optimiser(
        inputs=inputs,
        seed=random_seed,
    )

    edc = EDC(
        building_blocks=[
            "c_*x_",
            "c_*x_*x_",
            "c_*exp(c_ * x_)",
            "c_*x_**2",
        ],
        num_features=len(inputs),
        optimiser=optimiser,
        random_seed=random_seed,
        num_workers=args.num_workers,
        beam_width=10,
        max_depth=4,
        iterations=1000,
        search_algorithm=args.search,
    )

    splits = KFold(n_splits=10, shuffle=True, random_state=args.random_seed)

    data_object = {
        "elapsed_time": [],
        "auc_score": [],
        "accuracy_score": []
    }
    accuracy_scores = []
    auc_scores = []
    for index, (train_index, test_index) in enumerate(splits.split(X)):
        if args.fold is not None and args.fold != index:
            continue

        # Generate a new random seed
        random_seed = random.randrange(1, 2**32 - 1)
        logging.info(f"Random seed for fold: {random_seed}")
        random.seed(random_seed)
        optimiser.seed = random_seed

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        positive = np.count_nonzero(Y_test)
        baseline = max(positive / len(Y_test), 1 - positive / len(Y_test))
        if baseline == 1:
            logging.info("Baseline is 1, skipping fold")
            continue

        start_time = time.process_time()
        edc.fit(X_train, Y_train)
        end_time = time.process_time()
        elapsed_time = end_time - start_time

        logging.info(f"Best equation: {edc.best_equation}")
        logging.info(f"Best loss: {edc.best_loss}")
        logging.info(f"Elapsed time: {elapsed_time:.4f}s")


        auc_score = edc.auc_score(X_test, Y_test)
        accuracy_score = edc.accuracy_score(X_test, Y_test)

        logging.info(
            f"Baseline: {baseline:.4f}, AUC: {auc_score:.4f}, Accuracy: {accuracy_score:.4f}"
        )

        # Adding the results to the corresponding lists
        data_object["elapsed_time"].append(elapsed_time)
        data_object["auc_score"].append(auc_score)
        data_object["accuracy_score"].append(accuracy_score)

        if not args.discard_results:
            if args.fold is not None:
                filename = os.path.join(
                    args.output_folder,
                    f"{args.dataset}-{args.search}-{args.optimiser}-{args.random_seed}_{index}.json"
                )
            else:
                filename = os.path.join(
                    args.output_folder,
                    f"{args.dataset}-{args.search}-{args.optimiser}-{args.random_seed}.json"
                )
            with open(f"{filename}", "wb") as f:
                pickle.dump(data_object, f)

        if args.plot_decision_boundary:
            equation = str(edc.best_equation)
            # transform the equation to before normalisation
            for index, input_key in enumerate(inputs.keys()):
                equation = equation.replace(f"{input_key}", f"({input_key} * {scaler.scale_[index]} + {scaler.min_[index]})")
            equation = sympy.sympify(equation)

            # Calculate auc score and accuracy
            y_hat = sympy.lambdify(
                [sympy.symbols(key) for key in inputs],
                equation,
                "numpy",
            )(*scaler.inverse_transform(X_test).T)
            fpr, tpr, thresholds = roc_curve(Y_test, y_hat)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]

            plot_decision_boundary(scaler.inverse_transform(X_train), Y_train, equation, edc.best_loss, threshold)