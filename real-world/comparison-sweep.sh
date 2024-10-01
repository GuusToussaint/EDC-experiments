
# Create a list of datasets to evaluate
datasets=(
    "ADULT"
    "BANKNOTE"
    "BREAST"
    "CREDIT"
    "CYLINDER"
    "DIABETES"
    "HEPATITIS"
    "IONOSPHERE"
    "OCCUPANCY"
    "SONAR"
    "WISCONSIN"
)
classifiers=(
    "AMAXSC"
    "M4GP"
    "lda"
    "decision_tree"
    "svm_rbf"
    "random_forest"
    "MLP"
)

for clf in "${classifiers[@]}"
do
    for dataset in "${datasets[@]}"
    do
        sbatch start-comparison.slurm $dataset $clf
    done
done
