import os
import csv
import random
import pandas as pd

training_set = ""
test_set = ""
combined_set = ""
experiment_mode = ""

def train_test_split(X, y, test_size, y_regrouped, params):
    global training_set, test_set, combined_set, experiment_mode
    training_set, test_set, combined_set, experiment_mode = params['training_set'], params['test_set'], params['combined_set'], params['experiment_mode']

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


def get_files_in_directory(directory):
    file_list = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    
    return file_list


def is_empty_folder(path):
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path): 
            return True
        else:
            return False
    else:
        print("The provided path does not exist or is not a directory.")
        return True