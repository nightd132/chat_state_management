# Chat State Management


## Installation
Install the required dependencies:
```shell
pip install -r requirements.txt
```


## How to Run the Code
Run the scripts in the following order:
```shell
python -m src.experiment1
```
```shell
python -m src.autoencoder_train
```
```shell
python -m src.experiment2
```

## Workflow
1. experiment1.py
- Runs the first experiment.
- Generates intermediate outputs.
2. autoencoder_train.py
- Creates training data and train the Autoencoders for Experiment 2.
- The training data willbe store in autoencoder_training_data folder.
- The autoencoders will be store in autoencoders folder.
- Note: It takes around 40 mins for running in my device.
3. experiment2.py
- Uses the generated training Autoencoders to run the second experiment.
- Produces the final results.

Note: The scripts must be executed sequentially because each step depends on the outputs of the previous step.

## Results
- The result will be stored in results folder:
    - The experiment1 folder will store the result of experiment1 in csv file.
    - The experiment2 folder will store the result of experiment1 in csv file.
    - The plots folder will store the result plots:
        - The experiment1 folder will store the plots for experiment1.
        - The experiment2 folder will store the plots for experiment2.