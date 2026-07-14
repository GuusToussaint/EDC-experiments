
# Create a list of datasets to evaluate
datasets=(
    "sigma-0.0"
    "sigma-0.5"
    "sigma-1.0"
    "sigma-1.5"
    "sigma-2.0"
    "sigma-2.5"
    "sigma-3.0"
    "sigma-3.5"
    "sigma-4.0"
    "sigma-4.5"
    "sigma-5.0"
    "sigma-5.5"
    "sigma-6.0"
    "sigma-6.5"
    "sigma-7.0"
    "sigma-7.5"
    "sigma-8.0"
)
export random_seed=1805819

for dataset in "${datasets[@]}"
do
    sbatch start-artificial.slurm $dataset $random_seed
done
