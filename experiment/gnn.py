import os
from collections import Counter
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import csv
import chemprop
import random

experiment_name = ""
experiment_mode = ""
results_dir = ""
figures_dir = ""
combined_set = ""
training_set = ""
test_set = ""
chemprop_prediction = ""
chemprop_model_dir = ""
extract_file_from_hdf = ""
UNSOLVED_LENGTH = 0
NO_CUDA_OPTION = True
from experiment_config import Experiment

def set_parameters(experiment: Experiment):
    global experiment_name, experiment_mode, results_dir, figures_dir, combined_set, training_set, test_set, chemprop_prediction, chemprop_model_dir, extract_file_from_hdf, UNSOLVED_LENGTH, NO_CUDA_OPTION

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    results_dir = experiment.results_dir
    figures_dir = experiment.figures_dir
    combined_set = experiment.combined_set
    training_set = experiment.training_set
    test_set = experiment.test_set
    chemprop_prediction = experiment.chemprop_prediction
    chemprop_model_dir = experiment.chemprop_model_dir
    extract_file_from_hdf = experiment.extract_file_from_hdf
    UNSOLVED_LENGTH = experiment.UNSOLVED_LENGTH
    NO_CUDA_OPTION = experiment.NO_CUDA_OPTION


def is_empty_folder(path):
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path): 
            return True
        else:
            return False
    else:
        print("The provided path does not exist or is not a directory.")
        return True


def get_files_in_directory(directory):
    file_list = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    
    return file_list


def train_test_split(X, y, test_size, y_regrouped):
    if os.path.exists(training_set) and os.path.exists(test_set):
        print(f"Training and test sets already exist. Loading from files {training_set} and {test_set}...")
        # Read X_train, X_test, y_train, y_test from files
        with open(training_set, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            train_data = list(reader)
        with open(test_set, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            test_data = list(reader)
        
        # Separate X and y for train and test sets
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)
        
        return list(X_train), list(X_test), list(y_train), list(y_test)

    print(f"Training and test sets do not exist. Creating new training and test sets {training_set}, {test_set}...")
    if os.path.exists(combined_set):
        print(f"Combined set already exists. Loading from file {combined_set}...")
    else:
        combined = list(zip(X, y, y_regrouped))
        random.shuffle(combined)

        combined_dir = os.path.dirname(combined_set)
        if not os.path.exists(combined_dir):
            os.makedirs(combined_dir)

        with open(combined_set, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'route_length', 'route_length_regrouped'])  # Writing the header
            for row in combined:
                writer.writerow(row)
        

    combined_x_y = pd.read_csv(combined_set)

    if experiment_mode == 'multiclass':
        x_y = list(zip(combined_x_y['smiles'], combined_x_y['route_length_regrouped']))
    else:
        x_y = list(zip(combined_x_y['smiles'], combined_x_y['route_length']))
    
    # Split the data
    split_idx = int(len(x_y) * (1 - test_size))
    train_data = x_y[:split_idx]
    test_data = x_y[split_idx:]

    # Separating X and y for train and test sets
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    # Write to train.csv
    with open(training_set, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'route_length'])  # Writing the header
        for row in train_data:
            writer.writerow(row)

    # Write to test.csv
    with open(test_set, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'route_length'])  # Writing the header
        for row in test_data:
            writer.writerow(row)

    return list(X_train), list(X_test), list(y_train), list(y_test)

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

    X_train, X_test, y_train, y_test = train_test_split(raw_smiles, route_lengths, test_size=0.2, y_regrouped=route_lengths_regrouped)

    print(f"Performing {experiment_mode}. Train_test_split:\n X_train: ", len(X_train), ", X_test: ", len(X_test), ", y_train: ", len(y_train), ", y_test: ", len(y_test))

    if is_empty_folder(chemprop_model_dir):
        print("Model directory is empty - training new chemprop model...")
        if experiment_mode == 'multiclass':
            training_arguments = [
                '--data_path', training_set,
                '--dataset_type', 'multiclass',
                '--save_dir', chemprop_model_dir,
                '--multiclass_num_classes', num_classes,
            ]
        else:
            training_arguments = [
                '--data_path', training_set,
                '--dataset_type', experiment_mode,
                '--save_dir', chemprop_model_dir,
            ]
        if NO_CUDA_OPTION:
            training_arguments.append('--no_cuda')
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
    print(f"================================================== Finished GNN experiment {experiment_name} {experiment_mode}... ==================================================")


if __name__ == "__main__":
    run_gnn()
