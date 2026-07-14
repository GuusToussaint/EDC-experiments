
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
export random_seed=1805819

for fold in {0..9}
do
    for dataset in "${datasets[@]}"
    do
        sbatch start-edc.slurm $dataset $fold $random_seed
    done
done
