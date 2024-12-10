import os
import pickle
import pandas as pd

loss_types = ["auc", "hinge_loss"]
search_strategies = ["two_step", "normal"]

def load_files(dirname, search_approach, loss_function):
    results = []
    result_files = os.listdir(dirname)
    for result_file in result_files:
        dataset, search_strategy, optimiser, random_seed = result_file.split(
            "."
        )[0].split("-")
        with open(f"{dirname}/{result_file}", "rb") as f:
            res = pickle.load(f)
        for elapsed_time, auc_score, accuracy_score in zip(res["elapsed_time"], res["auc_score"], res["accuracy_score"]):
            results.append(
                [
                    dataset,
                    "EDC",  # "classifier
                    search_approach,
                    loss_function,
                    search_strategy,
                    optimiser,
                    elapsed_time,
                    auc_score,
                    accuracy_score,
                    random_seed,
                ]
            )
    return results

def get_dataframe():
    all_results = []
    for loss_type in loss_types:
        for search_strategy in search_strategies:
            results_folder = f"{loss_type}_{search_strategy}_results"
            results = load_files(results_folder, search_strategy, loss_type)
            all_results.extend(results)
    df = pd.DataFrame(
        all_results,
        columns=[
            "dataset",
            "classifier",
            "search_approach",
            "loss_function",
            "search_strategy",
            "optimiser",
            "elapsed_time",
            "auc_score",
            "accuracy_score",
            "random_seed",
        ],
    )

    return df

if __name__ == "__main__":
    results_df = get_dataframe()

    print(results_df.head())

    grouped_df = results_df.groupby(
        ["dataset", "classifier", "search_approach", "loss_function"]
    ).agg(
        {
            "elapsed_time": ["mean", "std"],
            "auc_score": ["mean", "std"],
            "accuracy_score": ["mean", "std"],
        }
    )

    for dataset in grouped_df.index.get_level_values("dataset").unique():
        print(f"{dataset}", end=" &")
        # For each combination of search_approach and loss_function print the mean and std of the auc_score
        results = {}
        for search_approach in search_strategies:
            for loss_function in loss_types:
                mean_auc = grouped_df.loc[(dataset, "EDC", search_approach, loss_function), "auc_score"]["mean"]
                std_auc = grouped_df.loc[(dataset, "EDC", search_approach, loss_function), "auc_score"]["std"]
                print(f"{mean_auc:.2f} $\pm$ {std_auc:.2f}", end=" &")
        print(f"\b \\\\")
