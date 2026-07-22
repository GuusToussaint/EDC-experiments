
# Create a list of datasets to evaluate
datasets=(
    "ADULT" # This takes to long
    # "BANKNOTE"
    # "BREAST" 
    # "CREDIT"
    # "CYLINDER"
    # "DIABETES"
    # "IONOSPHERE"
    # "OCCUPANCY"
    # "SONAR"
)
classifiers=(
    # "AMAXSC"
    # # "M4GP"
    # "lda"
    # "decision_tree"
    # "svm_rbf"
    # "random_forest"
    # "MLP"
    "PySR"
    "eggp"
)

export random_seed=1805819

for clf in "${classifiers[@]}"
do
    for dataset in "${datasets[@]}"
    do
        sbatch start-comparison.slurm $dataset $clf $random_seed
    done
done
