
# Create a list of datasets to evaluate
datasets=(
    "ADULT"
    # "BANKNOTE" # This is running
    # "BREAST" # This is running
    # "CREDIT" # This is running
    # "CYLINDER" # This is running
    # "DIABETES" # This is running
    # "HEPATITIS" # This is running
    # "IONOSPHERE" # This is running
    # "OCCUPANCY" # This is running
    # "SONAR" # This is running
    # "WISCONSIN" # This is running
)
export random_seed=1805819

for fold in {0..9}
do
    for dataset in "${datasets[@]}"
    do
        sbatch start-edc.slurm $dataset $fold $random_seed
    done
done
