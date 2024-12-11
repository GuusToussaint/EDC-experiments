import pickle
import pandas as pd
import numpy as np
import os

classifier_map = {
    "random_forest": "RF",
    "svm_rbf": "SVM",
    "decision_tree": "Tree",
    "lda": "LDA",
    "M4GP": "M4GP",
    "AMAXSC": "AMAXSC",
    "MLP": "MLP",
    "EDC": "EDC",
}

result_files = os.listdir("results")
results = []
for result_file in result_files:
    with open(os.path.join("results", result_file), "rb") as f:
        data_object = pickle.load(f)
        mean_auc = np.mean(data_object["auc_score"])
        dataset, classifier = result_file.split("-")[0], result_file.split("-")[1].split(".")[0]

        # add to a dataframe
        results.append((dataset, classifier, mean_auc))
df = pd.DataFrame(results, columns=["dataset", "classifier", "mean_auc"])

# print mean and std auc for each dataset and classifier
grouped_df = df.groupby(["classifier"]).agg({"mean_auc": ["mean", "std"]})

# Print the number of datasets for each classifier
for classifier in grouped_df.index:
    num_datasets = df[df["classifier"] == classifier].shape[0]
    print(f"{classifier_map[classifier]}&{num_datasets}\\\\")

print("\n")

# for each classifier print mean and std auc and sort by mean auc
grouped_df = grouped_df.sort_values(("mean_auc", "mean"), ascending=False)
for classifier in grouped_df.index:
    mean_auc = grouped_df.loc[classifier, ("mean_auc", "mean")]
    std_auc = grouped_df.loc[classifier, ("mean_auc", "std")]
    print(f"{classifier_map[classifier]}&{mean_auc:.2f}($\\pm {std_auc:.4f}$)\\\\")
