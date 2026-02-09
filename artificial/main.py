import os
import random
from EDC import EDC
from EDC.optimisers import HillClimberOptimiser
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import sympy
from EDC.util.load_data import load_csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

def normalise_data(X):
    return MinMaxScaler(feature_range=(-1, 1)).fit(X)

def plot_db(threshold, ax, X_plane, Y_plane, equation_lambda, color, linestyle, label):
    Z = equation_lambda(X_plane, Y_plane)
    ax.contour(
        X_plane,
        Y_plane,
        Z, 
        [threshold],
        colors=color,
        linewidths=2,
        linestyles=linestyle
    )
    ax.plot([], [], color=color, linewidth=3, label=label, linestyle=linestyle)

def evaluate_performance_for_dataset(X, Y, inputs, random_seed, num_workers, ground_truth_equation, equation_index, plot_decision_boundary=False):
    # Evaluate the performance of the model
    transformer = normalise_data(X)
    X = transformer.transform(X)

    optimiser = HillClimberOptimiser(
        fraction_random_sample=0.8307209522886011,
        step_size=0.29811510475390257,
        beam_width=2,
        inputs=inputs,
    )
    edc = EDC(
        building_blocks=[
            "c_*x_",
            "c_*x_*x_",
            "c_*exp(c_ * x_)",
            # "c_*x_**2",
        ],
        num_features=len(inputs),
        optimiser=optimiser,
        random_seed=random_seed,
        num_workers=num_workers,
        beam_width=10,
        max_depth=3,
        iterations=1000,
        search_algorithm="beam",
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_seed
    )

    # Calculate the auc and accuracy score for the ground truth equation
    y_hat = sympy.lambdify(
        [sympy.symbols(key) for key in inputs],
        ground_truth_equation,
        "numpy",
    )(*(transformer.inverse_transform(X_test)).T)
    ground_truth_auc_score = roc_auc_score(Y_test, y_hat)
    ground_truth_accuracy_score = accuracy_score(Y_test, y_hat >= 0)
    # plot_decision_boundary(transformer.inverse_transform(X_test), Y_test, ground_truth_equation, 0, save=True, filename=f"plots/{random_seed}_decision_boundary_ground_truth.png")
    
    # Train the model
    edc.fit(X_train, Y_train)

    # Calculate auc score and accuracy
    y_hat = sympy.lambdify(
        [sympy.symbols(key) for key in inputs],
        edc.best_equation,
        "numpy",
    )(*X_test.T)
    current_auc_score = roc_auc_score(Y_test, y_hat)

    fpr, tpr, thresholds = roc_curve(Y_test, y_hat)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]

    # plot_decision_boundary(X_test, Y_test, edc.best_equation, edc.best_loss, threshold, save=True, filename=f"plots/{random_seed}_decision_boundary_estimated.png")

    if plot_decision_boundary:

        ax = plt.figure(figsize=(8,8)).add_subplot()

        colors = ['black' if Y[i] else 'lightgray' for i in range(len(Y))]

        # Plot the data points
        original_X = transformer.inverse_transform(X)
        ax.scatter(original_X[:, 0], original_X[:, 1], marker="o", s=10, c=colors, alpha=1)

        # Plot the decision boundary
        X_plane = np.linspace(np.min(original_X[:, 0]), np.max(original_X[:, 0]), 10_000)
        Y_plane = np.linspace(np.min(original_X[:, 1]), np.max(original_X[:, 1]), 10_000)
        X_plane, Y_plane = np.meshgrid(X_plane, Y_plane)

        # plot ground truth decision boundary 
        ground_truth_lambda_equation = sympy.lambdify(
            [sympy.symbols(key) for key in inputs],
            ground_truth_equation,
            "numpy",
        )
        y_hat = ground_truth_lambda_equation(*transformer.inverse_transform(X_test).T)
        fpr, tpr, thresholds = roc_curve(Y_test, y_hat)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
        plot_db(
            threshold,
            ax,
            X_plane,
            Y_plane,
            ground_truth_lambda_equation,
            "red",
            "dashed",
            "Ground truth"
        )

        # plot estimated decision boundary
        equation = str(edc.best_equation)
        # transform the equation to before normalisation
        for index, input_key in enumerate(inputs.keys()):
            equation = equation.replace(f"{input_key}", f"({input_key} * {transformer.scale_[index]} + {transformer.min_[index]})")
        equation = sympy.sympify(equation)

        # Calculate auc score and accuracy
        estimated_equation_lambda = sympy.lambdify(
            [sympy.symbols(key) for key in inputs],
            equation,
            "numpy",
        )
        y_hat = estimated_equation_lambda(*transformer.inverse_transform(X_test).T)
        fpr, tpr, thresholds = roc_curve(Y_test, y_hat)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
        plot_db(
            threshold,
            ax,
            X_plane,
            Y_plane,
            estimated_equation_lambda,
            "blue",
            "solid",
            "Estimated"
        )

        # Show the plot
        ax.legend()
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        # plt.title(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{equation_index}_comparison.png')
        plt.clf()


    current_accuracy_score = accuracy_score(Y_test, y_hat >= threshold)

    return current_auc_score, current_accuracy_score, ground_truth_auc_score, ground_truth_accuracy_score

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimiser_logger = logging.getLogger("edc.optimisers")
    optimiser_logger.propagate = False

    random_seed = random.randrange(1, 2**32 - 1)

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The path to the data directory",
        choices=["artificial_random", "artificial_random_noise", "artificial_random_rich", "artificial_random_rich_noise"]
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
        "--num_workers",
        type=int,
        help="The number of workers to use for parallelisation",
        default=1,
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
        "--plot_decision_boundary",
        action="store_true",
        help="Whether to plot the decision boundary",
        default=False,
    )
    args = parser.parse_args()
    random_seed = args.random_seed
    logging.info(f"Random seed: {random_seed}")
    random.seed(random_seed)

    # Set the path to the data directory
    data_dir = os.path.join(
        args.dataset_folder,
        args.dataset,
    )

    data_object = {
        "auc_score": [],
        "accuracy_score": [],
        "ground_truth_auc_score": [],
        "ground_truth_accuracy_score": [],
    }
    data_files = os.listdir(data_dir)
    count = 0
    for index, data_file in enumerate(data_files):
        random_seed = random.randrange(1, 2**32 - 1)
        random.seed(random_seed)

        data_file_path = os.path.join(data_dir, data_file)
        X, Y, inputs = load_csv(data_file_path)

        ground_truth_equation = sympy.sympify('.'.join(data_file.split(".")[:-1]))

        auc, accuracy, ground_truth_auc, ground_truth_accuracy = evaluate_performance_for_dataset(
            X, 
            Y, 
            inputs, 
            random_seed=random_seed, 
            num_workers=args.num_workers, 
            ground_truth_equation=ground_truth_equation,
            equation_index=index,
            plot_decision_boundary=args.plot_decision_boundary
        )

        print(f"Index: {index}")
        print(f"AUC score: {auc}")
        print(f"Accuracy score: {accuracy}")
        print(f"Ground truth AUC score: {ground_truth_auc}")
        print(f"Ground truth accuracy score: {ground_truth_accuracy}")
        print("-"*25)

        data_object["auc_score"].append(auc)
        data_object["accuracy_score"].append(accuracy)
        data_object["ground_truth_auc_score"].append(ground_truth_auc)
        data_object["ground_truth_accuracy_score"].append(ground_truth_accuracy)


        if not args.discard_results:
            with open(os.path.join(args.output_folder, f"{args.dataset}-{args.random_seed}.json"), "wb") as f:
                pickle.dump(data_object, f)
