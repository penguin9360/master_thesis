import os
import csv
import random
import pandas as pd

training_set = ""
test_set = ""
validation_set = ""
combined_set = ""
experiment_mode = ""
inference_option = ""
inference_name = ""
inference_combined_set = ""
inference_test_set = ""

def check_and_remove_duplicates(X_train, X_test, y_train, y_test):
    # X_train_set = set(X_train)
    # y_train_set = set(y_train)
    X_test_filtered = []
    y_test_filtered = []
    duplicates_counter = 0
    for x, y in zip(X_test, y_test):
        if x not in X_train:
            X_test_filtered.append(x)
            y_test_filtered.append(y)
        else:
            duplicates_counter += 1
            print(f"Duplicate found: {x}, {y}")
    print(f"Removed {duplicates_counter} duplicated entries")
    return X_test_filtered, y_test_filtered
    

def train_test_split(X, y, test_size, val_size, y_regrouped, params):
    global training_set, test_set, validation_set, combined_set, experiment_mode, inference_option, inference_name, inference_combined_set, inference_test_set
    training_set, test_set, validation_set, combined_set, experiment_mode, inference_option, inference_name, inference_combined_set, inference_test_set = params['training_set'], params['test_set'], params['validation_set'], params['combined_set'], params['experiment_mode'], params['inference_option'], params['inference_name'], params['inference_combined_set'], params['inference_test_set']

    if os.path.exists(training_set) and os.path.exists(test_set) and os.path.exists(validation_set):
        print(f"Training, test, and validation sets already exist. Loading from files {training_set}, {test_set}, and {validation_set}...")
        # Read X_train, X_test, X_val, y_train, y_test, y_val from files
        with open(training_set, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            train_data = list(reader)
        with open(test_set, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            test_data = list(reader)
        with open(validation_set, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            val_data = list(reader)

        # Separate X and y for train and test sets
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)
        X_val, y_val = zip(*val_data)

        #!!!!!!!!
        if inference_option:
            X_test, y_test = check_and_remove_duplicates(X_train, X_test, y_train, y_test)
            X_val, y_val = check_and_remove_duplicates(X_train, X_val, y_train, y_val)
        
        return list(X_train), list(X_test), list(X_val), list(y_train), list(y_test), list(y_val)

    print(f"Training, test, or validation sets do not exist. Creating new training, test, and validation sets {training_set}, {test_set}, {validation_set}...")
    if os.path.exists(combined_set):
        print(f"Combined set already exists. Loading from file {combined_set}...")
    else:
        combined = list(zip(X, y, y_regrouped))
        random.shuffle(combined)
        print("First shuffle done")

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
    
    random.shuffle(x_y)
    print("Second shuffle done")
    
    # Split the data
    test_split_idx = int(len(x_y) * (1 - test_size))
    val_split_idx = int(len(x_y) * (1 - val_size - test_size))
    train_data = x_y[:val_split_idx]
    val_data = x_y[val_split_idx:test_split_idx]
    test_data = x_y[test_split_idx:]

    if inference_option:
        print(f"Inference test set path: {inference_test_set}")
        inf_test = pd.read_csv(inference_test_set)
        # combined_x_y_inference = pd.read_csv(inference_combined_set)
        # if experiment_mode == 'multiclass':
        #     x_y_inf = list(zip(combined_x_y_inference['smiles'], combined_x_y_inference['route_length_regrouped']))
        # else:
        #     x_y_inf = list(zip(combined_x_y_inference['smiles'], combined_x_y_inference['route_length']))
        test_data = list(zip(inf_test['smiles'], inf_test['route_length']))

        # split_idx = int(len(x_y_inf) * (1 - test_size))
        # test_data = random.shuffle(x_y_inf)[:split_idx]

        # test_data = x_y_inf[:split_idx]

    # Separating X and y for train and test sets
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    X_val, y_val = zip(*val_data)

    # handle duplicates here if inference option is enabled 
    # !!!!!!!
    if inference_option:
        X_test, y_test = check_and_remove_duplicates(X_train, X_test, y_train, y_test)

    if not os.path.exists(os.path.dirname(training_set)):
        os.makedirs(os.path.dirname(training_set))
    if not os.path.exists(os.path.dirname(test_set)):
        os.makedirs(os.path.dirname(test_set))
    if not os.path.exists(os.path.dirname(validation_set)):
        os.makedirs(os.path.dirname(validation_set))

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

    # Write to validation.csv
    with open(validation_set, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'route_length'])  # Writing the header
        for row in val_data:
            writer.writerow(row)

    return list(X_train), list(X_test), list(X_val), list(y_train), list(y_test), list(y_val)


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