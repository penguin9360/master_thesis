import os
import csv
import random
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator as ea
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns

training_set = ""
test_set = ""
validation_set = ""
combined_set = ""
experiment_name = ""
experiment_mode = ""
inference_option = ""
inference_name = ""
inference_combined_set = ""
inference_test_set = ""
figures_dir = ""
algorithm = ""


def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num - 1] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def write_hpo_slurm_file(hpo_slurm_file, experiment_name, experiment_mode):
    slurm_job_name = f"#SBATCH --job-name=HPO_{experiment_name}\n" # line 2
    if experiment_name == "50k":
        slurm_partition = "#SBATCH --partition=\"gpu-long\"\n" # line 3
        slurm_time = "#SBATCH --time=7-00:00:00\n" # line 4
    else:
        slurm_partition = "#SBATCH --partition=\"gpu-medium\"\n" # line 3
        slurm_time = "#SBATCH --time=1-00:00:00\n" # line 4
    slurm_output = f"#SBATCH --output=slurm_output/hpo/HPO_{experiment_name}_%A.out\n" # line 9

    replace_line(hpo_slurm_file, 2, slurm_job_name)
    replace_line(hpo_slurm_file, 3, slurm_partition)
    replace_line(hpo_slurm_file, 4, slurm_time)
    replace_line(hpo_slurm_file, 9, slurm_output)
    print(f"================================================== GNN HPO slurm file written for {experiment_name} {experiment_mode} ==================================================")
    print("Please use the command 'sbatch hpo.slurm' to run the GNN HPO experiment.")


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
    global training_set, test_set, validation_set, combined_set, experiment_name, experiment_mode, inference_option, inference_name, inference_combined_set, inference_test_set, figures_dir, algorithm
    training_set, test_set, validation_set, combined_set, experiment_name, experiment_mode, inference_option, inference_name, inference_combined_set, inference_test_set, figures_dir, algorithm = \
    params['training_set'], \
    params['test_set'], \
    params['validation_set'], \
    params['combined_set'], \
    params['experiment_name'], \
    params['experiment_mode'], \
    params['inference_option'], \
    params['inference_name'], \
    params['inference_combined_set'], \
    params['inference_test_set'], \
    params['figures_dir'], \
    params['algorithm']

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
    

# modified and improved based on: https://github.com/chingyaoc/Tensorboard2Seaborn/blob/master/beautify.py
def plot_tensorboard_learning_curves(params):
  ''' beautify tf log
      Use better library (seaborn) to plot tf event file'''
  print(f"plotting learning curves for {algorithm} {experiment_name} {experiment_mode}", "params: ", params)
  sns.set_theme(style="darkgrid")
  sns.set_context("paper")
  log_path = params['logdir']
  smooth_space = params['smooth']
  color_code = params['color']

  acc = ea.EventAccumulator(log_path, size_guidance={ea.SCALARS: 0})
  acc.Reload()

  # only support scalar now
  scalar_list = acc.Tags()['scalars']
  print("scalar_list: ", scalar_list)
  x_list = []
  y_list = []
  x_list_raw = []
  y_list_raw = []
  for tag in scalar_list:
    # print("tag: ", tag)
    # if 'param' in tag:
    #   for s in acc.Scalars(tag):
    #     print(f"s.step: {s.step}, s.value: {s.value}")
    x = [int(s.step) for s in acc.Scalars(tag)]
    y = [s.value for s in acc.Scalars(tag)]

    # smooth curve
    x_ = []
    y_ = []
    for i in range(0, len(x), smooth_space):
      if i + smooth_space >= len(x):
        x_.append(x[-1])
        y_.append(y[-1])
      else:
        x_.append(x[i])
        y_.append(sum(y[i:i+smooth_space]) / float(smooth_space))    
    x_list.append(x_)
    y_list.append(y_)

    # raw curve
    x_list_raw.append(x)
    y_list_raw.append(y)
    
    # fig, ax = plt.subplots()
    for i in range(len(x_list)):
      plt.figure(i)
      plt.clf()
      plt.subplot(111)
      plt.title(scalar_list[i])  
      plt.plot(x_list_raw[i], y_list_raw[i], color=colors.to_rgba(color_code, alpha=0.4))
      plt.plot(x_list[i], y_list[i], color=color_code, linewidth=1.5)
    
    plot_file = figures_dir + experiment_name + '_' + algorithm + '_' + experiment_mode + '_' + tag + '.png'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    plt.savefig(plot_file)