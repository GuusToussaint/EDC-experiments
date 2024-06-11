import pickle
import numpy as np
import argparse
import random
import logging
import os
from EDC.util import load_data
from EDC.util.load_data import load_csv
from EDC.optimisers import (
    RandomSampleOptimiser,
    GradientDecentOptimiser,
    RandomSearchOptimiser,
    SMACSearchOptimiser,
    HillClimberOptimiser,
    HillClimberGradientDecentOptimiser,
)
from functools import partial
from sklearn.calibration import expit
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import sympy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def normalise_data(X):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    return scaler.transform(X)

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
    random_seed = random.randrange(1, 2**32 - 1)

    parser = argparse.ArgumentParser(description="Compare optimisers")
    parser.add_argument(
        "--optimiser",
        type=str,
        help="The optimiser to use",
        required=True,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="The number of iterations to run the optimiser",
        default=100,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="The random seed to use",
        default=random_seed,
    )
    parser.add_argument(
        "--discard_results",
        action="store_true",
        help="Whether to store the results",
        default=False,
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="The folder containing the dataset files",
        required=True,
    )
    args = parser.parse_args()
    random_seed = args.random_seed
    logging.info(f"Random seed: {random_seed}")
    random.seed(random_seed)
    logging.info(args)

    optimiser_partial = get_optimiser(args.optimiser)

    data_object = {
        "auc_score": [],
        "accuracy_score": [],
    }
    for dataset_file in os.listdir(args.dataset_folder):
        random_seed = random.randrange(1, 2**32 - 1)
        logging.info(f"Random seed for fold: {random_seed}")

        equation = sympy.sympify('.'.join(dataset_file.split(".")[:-1]))
        numbers = equation.atoms(sympy.Number)
        number_counter = 0
        for index, number in enumerate(numbers):
            if number.is_integer:
                continue
            equation = equation.subs(number, sympy.symbols(f"c{number_counter}"))
            number_counter += 1

        X, Y, inputs = load_csv(os.path.join(args.dataset_folder, dataset_file))
        Y = Y.astype(int)
        X = normalise_data(X)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_seed
        )

        optimiser = optimiser_partial(
            inputs=inputs,
            X=X_train,
            Y=Y_train,
            seed=random_seed,
        )

        loss, config = optimiser.optimise(equation, args.iterations)

        # Calculate auc score and accuracy
        equation_with_best_config = equation.subs(config)
        y_hat = sympy.lambdify(
            [sympy.symbols(key) for key in inputs],
            equation_with_best_config,
            "numpy",
        )(*X_test.T)
        current_auc_score = roc_auc_score(Y_test, y_hat)

        fpr, tpr, thresholds = roc_curve(Y_test, y_hat)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

        current_accuracy_score = accuracy_score(Y_test, y_hat >= threshold)
        
        data_object["auc_score"].append(current_auc_score)
        data_object["accuracy_score"].append(current_accuracy_score)

        print(equation, X.shape, Y.shape)
        print(loss, current_auc_score, current_accuracy_score)

        if not args.discard_results:
            with open(f"results/{args.optimiser}-{args.random_seed}.json", "wb") as f:
                pickle.dump(data_object, f)
