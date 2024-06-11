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

## Error Handling

If the required arguments `--optimiser` and `--dataset_folder` are not provided, the script will raise an error. The error message will be similar to the following:

```sh
main.py: error: the following arguments are required: --optimiser, --dataset_folder
```

## Example Usage

### Running with all required arguments

```sh
python main.py --optimiser adam --dataset_folder /path/to/dataset
```

### Running with optional arguments

```sh
python main.py --optimiser sgd --iterations 1000 --random_seed 42 --discard_results --dataset_folder /path/to/dataset
```
