
# Create a list of datasets to evaluate
optimisers=(
    # "gradient_decent"
    # "random_sample"
    # "hill_climber"
    # "hill_climber_gradient_decent"
    "nelder_mead"
    "powell"
    "cg"
    # "bfgs"
    # "l_bfgs_b"
    # "tnc"
    # "cobyla"
    # "cobyqa"
    # "slsqp"
    # "trust_constr"

    # Don't work
    # "newton_cg"
    # "dogleg"
    # "trust_ncg"
    # "trust_exact"
    # "trust_krylov"
)
export random_seed=1805819

for optimiser in "${optimisers[@]}"
do
    sbatch start-optimisation.slurm $optimiser $random_seed
done
