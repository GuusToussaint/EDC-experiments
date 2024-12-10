import pandas as pd
import os
import pickle


if __name__ == "__main__":
    score_metric = "auc_score"
    score_metric_index = 0 if score_metric == "auc_score" else 1
    results = []

    result_files = os.listdir("results")
    for result_file in result_files:
        dataset, search_strategy, optimiser, random_seed = result_file.split(
            "."
        )[0].split("-")
        with open(f"results/{result_file}", "rb") as f:
            res = pickle.load(f)
        for elapsed_time, auc_score, accuracy_score in zip(res["elapsed_time"], res["auc_score"], res["accuracy_score"]):
            results.append(
                [
                    dataset,
                    "EDC-two-step",  # "classifier
                    search_strategy,
                    optimiser,
                    elapsed_time,
                    auc_score,
                    accuracy_score,
                    random_seed,
                ]
            )
    result_files = os.listdir("results-auc")
    for result_file in result_files:
        dataset, search_strategy, optimiser, random_seed = result_file.split(
            "."
        )[0].split("-")
        with open(f"results-auc/{result_file}", "rb") as f:
            res = pickle.load(f)
        for elapsed_time, auc_score, accuracy_score in zip(res["elapsed_time"], res["auc_score"], res["accuracy_score"]):
            results.append(
                [
                    dataset,
                    "EDC-normal",  # "classifier
                    search_strategy,
                    optimiser,
                    elapsed_time,
                    auc_score,
                    accuracy_score,
                    random_seed,
                ]
            )

    
    df = pd.DataFrame(
        columns=[
            "dataset",
            "classifier",
            "search_strategy",
            "optimiser",
            "elapsed_time",
            "auc_score",
            "accuracy_score",
            "random_seed"
        ],
        data=results,
    )


    datasets = [
        "BANKNOTE",
        "BREAST",
        "CREDIT",
        "DIABETES",
        "HEPATITIS",
        "IONOSPHERE",
        "OCCUPANCY",
        "SONAR",
        "WISCONSIN",
    ]

    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]

        # Get auc score for EDC-normal
        edc_normal_df = dataset_df[
            dataset_df["classifier"] == "EDC-normal"
        ]
        print(f"{dataset} EDC-normal AUC: {edc_normal_df['auc_score'].mean()}")

        # Get auc score for EDC-two-step
        edc_two_step_df = dataset_df[
            dataset_df["classifier"] == "EDC-two-step"
        ]
        print(f"{dataset} EDC-two-step AUC: {edc_two_step_df['auc_score'].mean()}")