import os
from collections import Counter
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import csv
import chemprop
from helpers import train_test_split, get_files_in_directory, is_empty_folder, plot_tensorboard_learning_curves
from experiment_config import Experiment
import argparse

experiment_name = ""
experiment_mode = ""
results_dir = ""
figures_dir = ""
combined_set = ""
training_set = ""
test_set = ""
validation_set = ""
chemprop_prediction = ""
chemprop_model_dir = ""
extract_file_from_hdf = ""
inference_option = ""
inference_name = ""
inference_combined_set = ""
inference_test_set = ""
UNSOLVED_LENGTH = 0
NO_CUDA_OPTION = True
model_parameters = []
tensorboard_log_dir = ""
graph_format_options = {}


def set_parameters(experiment: Experiment, model_param: list):
    global experiment_name, experiment_mode, results_dir, figures_dir, combined_set, training_set, test_set, validation_set, chemprop_prediction, chemprop_model_dir, extract_file_from_hdf, UNSOLVED_LENGTH, NO_CUDA_OPTION, inference_option, inference_name, inference_combined_set, inference_test_set, model_parameters, tensorboard_log_dir, graph_format_options

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    results_dir = experiment.results_dir
    figures_dir = experiment.figures_dir
    combined_set = experiment.combined_set
    training_set = experiment.training_set
    test_set = experiment.test_set
    validation_set = experiment.validation_set
    chemprop_prediction = experiment.chemprop_prediction
    chemprop_model_dir = experiment.chemprop_model_dir
    extract_file_from_hdf = experiment.extract_file_from_hdf
    UNSOLVED_LENGTH = experiment.UNSOLVED_LENGTH
    NO_CUDA_OPTION = experiment.NO_CUDA_OPTION
    inference_option = experiment.inference_option
    inference_name = experiment.inference_name
    inference_combined_set = experiment.inference_combined_set
    inference_test_set = experiment.inference_test_set
    model_parameters = model_param
    graph_format_options = experiment.graph_format_options

    tensorboard_log_dir = "./gnn/model/" + experiment_name + "_" + experiment_mode + "/fold_0/model_0" + "/"


def plot_gnn_learning_curves():
    for f in get_files_in_directory(tensorboard_log_dir):
        if '.pt' in f:
            continue
        print(f)
        parser = argparse.ArgumentParser()
        parser.add_argument('--logdir', default=f, type=str, help='logdir to event file')
        parser.add_argument('--smooth', default=25, type=float, help='window size for average smoothing')
        parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')

        args = parser.parse_args()
        params = vars(args)

        plot_tensorboard_learning_curves(params=params, epochs=model_parameters[1])


def run_gnn():
    # ================================================================== PROGRAM STARTS HERE ===================================================================
    print(f"================================================== Starting GNN experiment {experiment_name} {experiment_mode}... ==================================================")
    file_list = get_files_in_directory(results_dir)
    raw_smiles = []
    route_lengths = []
    unsolved_counter = 0

    # Read all HDF files and extract info
    if os.path.isfile(extract_file_from_hdf):
        print(f"Reading from HDF, extraction file exists. Loading from file {extract_file_from_hdf}...")
        with open(extract_file_from_hdf, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            combined_data = list(reader)
            raw_smiles, route_lengths = zip(*combined_data)
        for i in range(len(route_lengths)):
            if int(route_lengths[i]) == 0:
                unsolved_counter += 1
    else:
        print(f"Reading from HDF and writing to extraction file {extract_file_from_hdf}...")
        for f in file_list:
            data = pd.read_hdf(f, "table")
            for i in range(len(data)):
                smile = data['target'][i]
                is_solved = data['is_solved'][i]
                route_length = data['number_of_steps'][i]
                if not is_solved:
                    unsolved_counter += 1
                    print(f"Unsolved molecule #{unsolved_counter}: {smile}, with original route length {route_length} set to {UNSOLVED_LENGTH}")
                    route_length = UNSOLVED_LENGTH
                raw_smiles.append(smile)
                route_lengths.append(route_length)
                combined_data = list(zip(raw_smiles, route_lengths))

        with open(extract_file_from_hdf, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'route_length'])
            writer.writerows(combined_data)

    print(f"       =================================================================================================================================================== \n\
        Import completed. {len(raw_smiles) - unsolved_counter} out of {len(raw_smiles)} were solved. Starting encoding... \n \
        ===================================================================================================================================================")

    # regroup route lengths into 4 classes -- for multiclass tasks
    # 1. 1-2 steps
    # 2. 3-5 steps
    # 3. 6+ steps
    # 0. unsolved (length == 0 as specified earlier)

    route_lengths_regrouped = []
    num_classes = "4"

    for i in route_lengths:
        if int(i) == 1 or int(i) == 2:
            route_lengths_regrouped.append("1")
        if int(i) >= 3 and int(i) <= 5:
            route_lengths_regrouped.append("2")
        if int(i) >= 6:
            route_lengths_regrouped.append("3")
        if int(i) == 0:
            route_lengths_regrouped.append("0")

    print(f"Regrouped route lengths:\n {Counter(route_lengths_regrouped).keys()} \n {Counter(route_lengths_regrouped).values()}")
    
    params = {
        'training_set': training_set,
        'test_set': test_set,
        'validation_set': validation_set,
        'combined_set': combined_set,
        'experiment_name': experiment_name,
        'experiment_mode': experiment_mode,
        'inference_option': inference_option,
        'inference_name': inference_name,
        'inference_combined_set': inference_combined_set,
        'inference_test_set': inference_test_set, 
        'figures_dir': figures_dir,
        'algorithm': "gnn",
    }

    X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(raw_smiles, route_lengths, test_size=0.1, val_size=0.1, y_regrouped=route_lengths_regrouped, params=params)

    print(f"Performing {experiment_mode}. Train_test_split:\n X_train: ", len(X_train), ", X_test: ", len(X_test), ", X_val: ", len(X_val), ", y_train: ", len(y_train), ", y_test: ", len(y_test), ", y_val: ", len(y_val))

    if is_empty_folder(chemprop_model_dir):
        print("Model directory is empty - training new chemprop model...")
        if experiment_mode == 'multiclass':
            training_arguments = [
                '--data_path', training_set,
                '--dataset_type', 'multiclass',
                '--save_dir', chemprop_model_dir,
                '--multiclass_num_classes', num_classes,
                '--loss_function', 'cross_entropy',
                '--metric', 'cross_entropy',
                '--extra_metrics', 'accuracy', "f1",
            ]
        else:
            training_arguments = [
                '--data_path', training_set,
                '--dataset_type', experiment_mode,
                '--save_dir', chemprop_model_dir,
                '--loss_function', 'mse', # mse is the default loss function
                '--metric', 'rmse',
                '--extra_metrics', 'r2'
            ]
        if NO_CUDA_OPTION:
            training_arguments.append('--no_cuda')

        training_arguments_with_validation = ["--separate_val_path", validation_set, "--separate_test_path", test_set]
        training_arguments += training_arguments_with_validation
        training_arguments += model_parameters
        
        args = chemprop.args.TrainArgs().parse_args(training_arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
        print("Chemprop training score: \nMean: ", mean_score, " | std: ", std_score)

    print("Running predictions using chemprop model in ", chemprop_model_dir)
    predicting_arguments = [
        '--test_path', test_set,
        '--preds_path', chemprop_prediction,
        '--checkpoint_dir', chemprop_model_dir,
    ]
    if NO_CUDA_OPTION:
        predicting_arguments.append('--no_cuda')
    args = chemprop.args.PredictArgs().parse_args(predicting_arguments)
    preds = chemprop.train.make_predictions(args=args)

    print("Predictions completed. Writing to file ", chemprop_prediction)

    if not inference_option:
        print("Plotting GNN learning curves...")
        print(f"files in tensorboard dir {tensorboard_log_dir}: ", get_files_in_directory(tensorboard_log_dir))
        plot_gnn_learning_curves()

    print(f"================================================== Finished GNN experiment {experiment_name} {experiment_mode}... ==================================================")


if __name__ == "__main__":
    run_gnn()
