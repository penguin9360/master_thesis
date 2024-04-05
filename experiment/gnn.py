import os
from collections import Counter
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import csv
import chemprop
import random

# experiment_name = "50ktest2"
experiment_name = "1k"
experiment_mode = 'regression' # 'regression' or 'multiclass' 
results_dir = "results/" + experiment_name + "/"
figures_dir = "figures/" + experiment_name + "/"
training_set = "./train_test/" + experiment_name + "_" + experiment_mode + "_train.csv"
test_set = "./train_test/" + experiment_name + "_" + experiment_mode + "_test.csv"
chemprop_prediction = "gnn/data/" + experiment_name + "_" + experiment_mode + "_prediction"
chemprop_model_dir = "gnn/model/" + experiment_name + "_" + experiment_mode 
extract_file_from_hdf = "./result_extract/" + experiment_name + "_extract.csv"
UNSOLVED_LENGTH = 0
NO_CUDA_OPTION = 'False'

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
    # Combine X and y to shuffle together
    combined = list(zip(X, y))
    #@TODO: Buggy, figure out a way to accomodate two data types for two experiment modes
    if 'multiclass' in training_set:
        combined = list(zip(X, y_regrouped))
    random.shuffle(combined)  # Shuffling the combined data

    # Split the data
    split_idx = int(len(combined) * (1 - test_size))
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    # Separating X and y for train and test sets
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    # Write to train.csv
    with open(training_set, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'route_length'])  # Writing the header
        for row in train_data:
            writer.writerow(row)
            
    
    # Write to multiclass training set
    multiclass_training_set = training_set.replace("_regression_", "_multiclass_")
    with open(multiclass_training_set, mode='w', newline='') as file:
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

    # Write to multiclass test set
    multiclass_test_set = test_set.replace("_regression_", "_multiclass_")
    with open(multiclass_test_set, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'route_length'])
        for row in test_data:
            writer.writerow(row)

    return list(X_train), list(X_test), list(y_train), list(y_test)
    # return X_train, X_test, y_train, y_test

# ================================================================== PROGRAM STARTS HERE ===================================================================

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
# 4. unsolved (length == 0 as specified earlier)

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

# to do multiclass, change route_lengths to route_lengths_regrouped
if experiment_mode == 'regression':
    X_train, X_test, y_train, y_test = train_test_split(raw_smiles, route_lengths, test_size=0.2, y_regrouped=route_lengths_regrouped)
else:
    X_train, X_test, y_train, y_test = train_test_split(raw_smiles, route_lengths_regrouped, test_size=0.2, y_regrouped=route_lengths_regrouped)
print(f"Performing {experiment_mode}. Train_test_split:\n X_train: ", len(X_train), ", X_test: ", len(X_test), ", y_train: ", len(y_train), ", y_test: ", len(y_test))

# print("Preparing training data for chemprop...")
# with open(training_set, 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['smile','route_length'])
#     rows = []
#     for index, value in enumerate(X_train):
#         rows.append([str(value), str(y_train[index])])
#     writer.writerows(rows)
# print("Training Data preparation for chemprop completed.")

# print("Preparing testing data for chemprop...")
# with open(test_set, 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['smile','route_length'])
#     rows = []
#     for index, value in enumerate(X_test):
#         rows.append([str(value), str(y_test[index])])
#     writer.writerows(rows)
# print("Testing Data preparation for chemprop completed.")

if is_empty_folder(chemprop_model_dir):
    print("Model directory is empty - training new chemprop model...")
    if experiment_mode == 'multiclass':
        training_arguments = [
            '--data_path', training_set,
            '--dataset_type', 'multiclass',
            '--save_dir', chemprop_model_dir,
            # '--no_cuda', NO_CUDA_OPTION,
            '--multiclass_num_classes', num_classes,
        ]
    else:
        training_arguments = [
            '--data_path', training_set,
            '--dataset_type', experiment_mode,
            '--save_dir', chemprop_model_dir,
            # '--no_cuda', NO_CUDA_OPTION,
        ]
    args = chemprop.args.TrainArgs().parse_args(training_arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    print("Chemprop training score: \nMean: ", mean_score, " | std: ", std_score)

print("Running predictions using chemprop model in ", chemprop_model_dir)
predicting_arguments = [
    '--test_path', test_set,
    '--preds_path', chemprop_prediction,
    '--checkpoint_dir', chemprop_model_dir,
]
args = chemprop.args.PredictArgs().parse_args(predicting_arguments)
preds = chemprop.train.make_predictions(args=args)