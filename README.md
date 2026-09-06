# Chat State Management


## Installation
The project is tested with Python 3.11. A CUDA-capable Linux environment is recommended because the default configuration uses the Mamba2 model with `cuda` and `bfloat16`. The experiment is run on Nvidia RTX6000	24GB at first but for some reason (someone use this) I had to change to Nvidia Tesla A100 80 GB (my uni gpu servers).

Create and activate a virtual environment:
```shell
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:
```shell
pip install -r requirements.txt
```

## How to Run the Code
Run the initial experiments with:
```shell
python -m src.experiment1
```

Train the autoencoders required by Experiment 2:
```shell
python -m src.autoencoder_train
```

Then run the compression comparison:
```shell
python -m src.experiment2
```

The state carry-over experiments can be run independently after installing the dependencies:
```shell
python -m src.experiment3
python -m src.experiment4
```

Experiments 1, 3, and 4 require only the configured model and dataset.
Experiment 2 additionally requires the autoencoder checkpoints produced by
`src.autoencoder_train`.

## Workflow
1. experiment1.py
- Runs the first experiment.
- Compares full-history baseline inference with recurrent-state inference.
2. autoencoder_train.py
- Creates training data and train the Autoencoders for Experiment 2.
- Training data is stored in `autoencoder_training_data/`.
- Autoencoder checkpoints are stored in `autoencoders/`.
3. experiment2.py
- Uses the generated autoencoder checkpoints to compare compression methods.
4. experiment3.py
- Evaluates forgetting-factor state carry-over across session boundaries.
5. experiment4.py
- Evaluates EMA state carry-over across session boundaries.

Only Experiment 2 depends on a previous script. Experiments 1, 3, and 4 can
be run separately. Experiments 3 and 4 cache chain results under each
experiment's `results/.../cache/` directory; rerun them after changing the
model, dataset, configuration, or source code, and remove the affected cache
files when necessary. Experiment 4 also supports `force_rerun_labels` in its
entry point.

## Results
Results are stored under `results/`:
- `results/experiment1/experiment1.csv` contains paired baseline/state PPL, latency, text-history size, state size, and comparison metrics.
- `results/experiment2/experiment2.csv` contains compression-method comparisons and their PPL, latency, and storage metrics.
- `results/experiment3/experiment3.csv` contains forgetting-factor summary metrics, and `experiment3_boundary_sequence.csv` contains boundary values.
- `results/experiment4/experiment4.csv` contains EMA summary metrics, and `experiment4_boundary_sequence.csv` contains boundary values.
- `results/plots/experiment1/`, `results/plots/experiment2/`, `results/plots/experiment3/`, and `results/plots/experiment4/` contain generated figures.

Common metrics are measured per turn or boundary: PPL is perplexity, latency is elapsed inference time in seconds, and text/state size is the serialized storage size in KB. Experiment 2 additionally reports compression and speed comparisons between the uncompressed baseline and each method.