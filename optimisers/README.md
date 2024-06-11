# Optimiser experiments

This folder contains the code to run the optimiser experiments described in the paper.
## Usage

To run the script, use the following command:

```sh
python main.py --optimiser OPTIMISER [--iterations ITERATIONS] [--random_seed RANDOM_SEED] [--discard_results] --dataset_folder DATASET_FOLDER
```

## Required Arguments

- `--optimiser OPTIMISER`: Specifies the optimizer to be used. This is a mandatory argument.
- `--dataset_folder DATASET_FOLDER`: Specifies the path to the dataset folder. This is a mandatory argument.

## Optional Arguments

- `-h, --help`: Shows the help message and exits.
- `--iterations ITERATIONS`: Specifies the number of iterations for the optimization process. Default is set to a specific value if not provided.
- `--random_seed RANDOM_SEED`: Sets the random seed for reproducibility of results. Default is set to a specific value if not provided.
- `--discard_results`: If specified, the results will be discarded after processing.

## Example Usage

### Running for the hill climber optimiser

```sh
python main.py --dataset_folder ~/code/EDC-datasets/artificial_random --optimiser hill_climber --iterations 1000
```

### Running for the random sample optimiser

```sh
python main.py --dataset_folder ~/code/EDC-datasets/artificial_random --optimiser random_sample --iterations 1000
```

### Running for the gradient decent + random sample optimiser

```sh
python main.py --dataset_folder ~/code/EDC-datasets/artificial_random --optimiser gradient_decent --iterations 1000
```

### Running for the gradient decent + hill climber optimiser

```sh
python main.py --dataset_folder ~/code/EDC-datasets/artificial_random --optimiser hill_climber_gradient_decent --iterations 1000
```
