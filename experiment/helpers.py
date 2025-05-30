import os
import csv
import random
import string
import pandas as pd
import numpy as np
from experiment_config import Experiment
import subprocess as subp
from time import sleep

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
graph_format_options = {}


def set_parameters(experiment: Experiment):
    global experiment_name, experiment_mode, results_dir, figures_dir, file_name, test_file, xgboost_prediction, test_set, inference_option, inference_name, graph_format_options

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    results_dir = experiment.results_dir
    figures_dir = experiment.figures_dir
    file_name = experiment.chemprop_prediction
    test_file = experiment.test_set
    xgboost_prediction = experiment.xgboost_prediction
    test_set = experiment.test_set
    inference_option = experiment.inference_option
    inference_name = experiment.inference_name
    graph_format_options = experiment.graph_format_options

    if inference_option:
        experiment_name += "_inference_" + inference_name


def append_lines(file_name, lines_to_add):
    with open(file_name, 'a') as f:
        if isinstance(lines_to_add, str):
            f.write('\n' + lines_to_add)
        else:
            for line in lines_to_add:
                f.write('\n' + line)


def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num - 1] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def submit_hpo_slurm(hpo_slurm_file, experiment_name, experiment_mode, search_option, num_evals):
    slurm_job_name = f"#SBATCH --job-name=HPO_{experiment_name}_{experiment_mode}_{search_option}_{num_evals}\n" # line 2
    if experiment_name == "50k" or experiment_name == "10k":
        slurm_partition = "#SBATCH --partition=\"gpu-long\"\n" # line 3
        slurm_time = "#SBATCH --time=7-00:00:00\n" # line 4
    else:
        slurm_partition = "#SBATCH --partition=\"gpu-medium\"\n" # line 3
        slurm_time = "#SBATCH --time=1-00:00:00\n" # line 4
    slurm_output = f"#SBATCH --output=slurm_output/hpo/HPO_{experiment_name}_{experiment_mode}_{search_option}_{num_evals}_%A.out\n" # line 9

    replace_line(hpo_slurm_file, 2, slurm_job_name)
    replace_line(hpo_slurm_file, 3, slurm_partition)
    replace_line(hpo_slurm_file, 4, slurm_time)
    replace_line(hpo_slurm_file, 9, slurm_output)
    print(f"================================================== GNN HPO slurm file written for {experiment_name} {experiment_mode} {search_option} search with num_evals = {num_evals} ==================================================")
    if experiment_mode == 'regression':
        print("Job submitted as specifed in 'hpo_regression.slurm'. If rerun is needed, use the command 'sbatch hpo_regression.slurm' to run the GNN HPO experiment.")
    if experiment_mode == 'multiclass':
        print("Job submitted as specifed in 'hpo_multiclass.slurm'. If rerun is needed, use the command 'sbatch hpo_multiclass.slurm' to run the GNN HPO experiment.")
    subp.check_call(['sbatch', hpo_slurm_file])
    print("====================================================== current queue: =======================================================")
    subp.check_call(['squeue', '--me'])
    # sleep(3) # very important as otherwise the slurm jobs cant distinguish log directories and will overwrite the output files


def submit_offline_experiment_slurm(offline_slurm_file, offline_start_file):
    slurm_job_name = "#SBATCH --job-name=%s\n" % str(offline_start_file).split(".")[0]  # line 2
    if str(offline_start_file).startswith("50k"):
        slurm_partition = "#SBATCH --partition=\"gpu-long\"\n" # line 3
        slurm_time = "#SBATCH --time=7-00:00:00\n" # line 4
    elif str(offline_start_file).startswith("1k"):
        slurm_partition = "#SBATCH --partition=\"gpu-short\"\n"
        slurm_time = "#SBATCH --time=04:00:00\n"
    else:
        slurm_partition = "#SBATCH --partition=\"gpu-medium\"\n" # line 3
        slurm_time = "#SBATCH --time=1-00:00:00\n" # line 4
    slurm_output = "#SBATCH --output=slurm_output/offline_run/%s_%%A.out\n" % str(offline_start_file).split(".")[0]  # line 9

    replace_line(offline_slurm_file, 2, slurm_job_name)
    replace_line(offline_slurm_file, 3, slurm_partition)
    replace_line(offline_slurm_file, 4, slurm_time)
    replace_line(offline_slurm_file, 9, slurm_output)
    replace_line(offline_slurm_file, 14, f"python {offline_start_file}\n")
    print(f"================================================== Offline experiment slurm file written and submitted for {offline_start_file}  ==================================================")
    subp.check_call(['sbatch', offline_slurm_file])
    print("====================================================== current queue: =======================================================")
    subp.check_call(['squeue', '--me'])
    # sleep(3) # very important as otherwise the slurm jobs cant distinguish log directories and will overwrite the output files


def generate_random_string(length=6):
            characters = string.ascii_lowercase + string.digits
            return ''.join(random.choice(characters) for _ in range(length))


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
    

def plot_xgboost_learning_curves(experiment_mode, model, metrics, save_dir, epochs):
    smooth_space = 5
    color_code = graph_format_options['training_graph_line_color']
    for metric in metrics:
        print(f"plotting learning curves for {algorithm} {experiment_name} {experiment_mode} {metric}")
        loss_plot_path = figures_dir + experiment_name + "_xg_boost_" + experiment_mode + "_validation_" + metric + "_plot.png"
        fig, ax = plt.subplots(figsize=graph_format_options['default_plot_size'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        results = model.evals_result()
        
        if experiment_mode == 'regression':
            if 'logloss' in metric:
                continue
            # Get raw values
            y_raw = results['validation_1'][metric]
            x_raw = list(range(len(y_raw)))
            
            # Calculate smooth curve
            x_smooth = []
            y_smooth = []
            for i in range(0, len(x_raw), smooth_space):
                if i + smooth_space >= len(x_raw):
                    x_smooth.append(x_raw[-1])
                    y_smooth.append(y_raw[-1])
                else:
                    x_smooth.append(x_raw[i])
                    y_smooth.append(sum(y_raw[i:i+smooth_space]) / float(smooth_space))
            
            if graph_format_options['smooth_curve_option']:
                ax.plot(x_raw, y_raw, color=colors.to_rgba(color_code, alpha=0.4), label='regression_train_raw')
                ax.plot(x_smooth, y_smooth, color=color_code, linewidth=1.5, label='regression_train_smooth')
            else:
                ax.plot(x_raw, y_raw, color=color_code, label='regression_train_raw')
        else:
            if 'rmse' in metric:
                continue
            # Get raw values
            y_raw = results['validation_1'][metric]
            x_raw = list(range(len(y_raw)))
            
            # Calculate smooth curve
            x_smooth = []
            y_smooth = []
            for i in range(0, len(x_raw), smooth_space):
                if i + smooth_space >= len(x_raw):
                    x_smooth.append(x_raw[-1])
                    y_smooth.append(y_raw[-1])
                else:
                    x_smooth.append(x_raw[i])
                    y_smooth.append(sum(y_raw[i:i+smooth_space]) / float(smooth_space))
            
            # Plot both raw and smooth curves
            if graph_format_options['smooth_curve_option']:
                ax.plot(x_raw, y_raw, color=colors.to_rgba(color_code, alpha=0.4), label='multiclass_train_raw')
                ax.plot(x_smooth, y_smooth, color=color_code, linewidth=1.5, label='multiclass_train_smooth')
            else:
                ax.plot(x_raw, y_raw, color=color_code, label='multiclass_train_raw')
        
        plt.ylabel("validation_" + metric, fontsize=graph_format_options['label_font_size'])
        plt.xlabel(graph_format_options['training_graph_xlabel'], fontsize=graph_format_options['label_font_size'])

        if 'logloss' in metric:
            if experiment_mode == 'multiclass':
                plt.ylim(graph_format_options['train_loss_ylim'])
                plt.ylabel("validation_cross_entropy", fontsize=graph_format_options['label_font_size'])
            plt.ylim(graph_format_options['crossentropy_ylim'])
        
        if 'rmse' in metric:
            plt.ylim(graph_format_options['rmse_ylim'])

        # tick magic
        x_positions = np.linspace(0, epochs, graph_format_options['training_graph_num_xticks'])  
        x_labels = [f"{int(e)}" for e in x_positions]
        y_min, y_max = plt.ylim()
        # y_ticks = np.linspace(y_min, y_max, graph_format_options['training_graph_num_yticks'])
        # plt.yticks(y_ticks)
        plt.xticks(x_positions, x_labels, fontsize=graph_format_options['tick_font_size'])
        plt.yticks(rotation=graph_format_options['training_graph_ytick_rotation'])
        plt.grid(True, color=graph_format_options['grid_color'])
        plt.gca().set_facecolor(graph_format_options['training_graph_background_color'])
        plt.yticks(fontsize=graph_format_options['tick_font_size'], rotation=graph_format_options['training_graph_ytick_rotation'])
        
        plt.gca().spines['top'].set_visible(False)   
        plt.gca().spines['right'].set_visible(False) 
        plt.gca().spines['bottom'].set_visible(False) 
        plt.gca().spines['left'].set_visible(False) 
        plt.savefig(loss_plot_path)


# modified and improved based on: https://github.com/chingyaoc/Tensorboard2Seaborn/blob/master/beautify.py
def plot_tensorboard_learning_curves(params, epochs):
  print(f"plotting learning curves for {algorithm} {experiment_name} {experiment_mode}", "params: ", params, "epochs: ", epochs)
  epochs = int(epochs)
  sns.set_theme(style="darkgrid")
  sns.set_context("paper")
  log_path = params['logdir']
  smooth_space = params['smooth']
  color_code = graph_format_options['training_graph_line_color']

  acc = ea.EventAccumulator(log_path, size_guidance={ea.SCALARS: 0})
  acc.Reload()

#   print("Debug - Full Tags:", acc.Tags())  # Check all available tags
#   print("Debug - Available keys:", acc.Tags().keys())  # Check if 'scalars' exists

  # only support scalar now
  scalar_list = acc.Tags()['scalars']
  scalar_list = [tag for tag in scalar_list if 'test' not in tag and 'norm' not in tag]
  print("Debug - Scalar list:", scalar_list)  # Check what's in scalar_list
  
  x_list = []
  y_list = []
  x_list_raw = []
  y_list_raw = []
  total_steps = 0
  for tag in scalar_list:
    # if 'param' in tag:
    #   for s in acc.Scalars(tag):
    #     print(f"s.step: {s.step}, s.value: {s.value}")
    x = [int(s.step) for s in acc.Scalars(tag)]
    y = [s.value for s in acc.Scalars(tag)]
    total_steps = max(total_steps, x[-1])

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

    current_idx = len(x_list_raw) - 1  # Index of current tag's data

    print(f"Debug - Plotting {tag} with {len(x_list_raw[current_idx])} points")
    # print(f"Debug - x range: {min(x_list_raw[current_idx])} to {max(x_list_raw[current_idx])}")
    # print(f"Debug - y range: {min(y_list_raw[current_idx])} to {max(y_list_raw[current_idx])}")

    plt.figure(current_idx, figsize=graph_format_options['default_plot_size'])
    plt.clf()
    plt.subplot(111)
    if graph_format_options['smooth_curve_option']:
        plt.plot(x_list_raw[current_idx], y_list_raw[current_idx], color=colors.to_rgba(color_code, alpha=0.4))
        plt.plot(x_list[current_idx], y_list[current_idx], color=color_code, linewidth=1.5)
    else:
        plt.plot(x_list_raw[current_idx], y_list_raw[current_idx], color=color_code)
    plt.xlabel(graph_format_options['training_graph_xlabel'], fontsize=graph_format_options['label_font_size'])
    plt.ylabel(scalar_list[current_idx], fontsize=graph_format_options['label_font_size'])

    if 'entropy' in tag:
        plt.ylim(graph_format_options['crossentropy_ylim'])
    
    if 'rmse' in tag:
        plt.ylim(graph_format_options['rmse_ylim'])

    # if 'validation' in tag or 'test' in tag:
    #     # plt.yscale('log')
        
    #     ax = plt.gca()
    #     ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    #     ax.yaxis.set_minor_formatter(plt.ScalarFormatter(useOffset=False))
    #     ax.yaxis.set_major_locator(plt.LogLocator(numticks=graph_format_options['training_graph_num_xticks']))
        
    if 'train_loss' in tag:
        plt.ylim(graph_format_options['train_loss_ylim'])
        plt.axhline(y=0, color='lightgrey', linestyle='--', zorder=1)

    if 'learning_rate' in tag:
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.grid(True)
    
    # tick magic to convert xaxis numbers from steps to epochs
    num_ticks = graph_format_options['training_graph_num_xticks']
    x_steps = np.linspace(0, total_steps, num_ticks)
    x_epochs = np.linspace(0, epochs, num_ticks)

    plt.xticks(x_steps, [f"{int(e)}" for e in x_epochs], fontsize=graph_format_options["tick_font_size"])
    plt.yticks(fontsize=graph_format_options["tick_font_size"])
    if not 'validation' in tag:
        plt.yticks(rotation=graph_format_options['training_graph_ytick_rotation'])

    plot_file = figures_dir + experiment_name + '_' + algorithm + '_' + experiment_mode + '_' + tag + '.png'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    plt.savefig(plot_file)


def make_box_plot(df, metrics: list, algorithm):
    rmse, mae, r_squared = metrics[0:3]
    box_plot_path = figures_dir + experiment_name + "_" + algorithm + "_" + experiment_mode + "_box_plot.png"
    box_plot_title = experiment_name + "_" + algorithm + "_" + experiment_mode + "_box_plot"
    print(f"plotting {box_plot_title}...")
    fig, ax = plt.subplots(figsize=graph_format_options['default_plot_size'])
    ax.plot([0, 10], [0, 10], '--', lw=1, color='lightgrey', zorder=1)
    sns.boxplot(
        x=df['route_length_truth'],
        y=df['route_length_predicted'],
        showfliers=False,
        zorder=2,
        ax=ax
    )
    ax.set_xlim(graph_format_options['box_plot_xlim'])
    ax.set_ylim(graph_format_options['box_plot_ylim'])
    ax.set_xlabel('True Route Length', fontsize=graph_format_options['label_font_size'])
    ax.set_ylabel('Predicted Route Length', fontsize=graph_format_options['label_font_size'])
    ax.set_title(f"\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.2f}", fontsize=graph_format_options['label_font_size'])
    ax.tick_params(axis='both', which='major', labelsize=graph_format_options['tick_font_size'])
    x_positions = ax.get_xticks()
    xticks_labels = [str(int(x)) for x in x_positions]
    xticks_labels[0] = 'unsolved'
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticks_labels)
    print(f"Saving box plot to {box_plot_path}")
    os.makedirs(os.path.dirname(box_plot_path), exist_ok=True)
    plt.savefig(box_plot_path)


def make_reg_plot(df, metrics: list, algorithm):
    rmse, mae, r_squared = metrics[0:3]
    reg_plot_path = figures_dir + experiment_name + "_" + algorithm + "_" + experiment_mode + "_regression_plot.png"
    reg_plot_title = experiment_name + "_" + algorithm + "_" + experiment_mode + "_regression_plot"
        
    print(f"plotting {reg_plot_title}...")
        
    fig, ax = plt.subplots(figsize=graph_format_options['default_plot_size'])
    ax.plot([0, 10], [0, 10], '--', lw=1, color='lightgrey')
    sns.regplot(
        x=df['route_length_truth'],
        y=df['route_length_predicted'],
        ax=ax
    )
    ax.set_xlim(graph_format_options['box_plot_xlim'])
    ax.set_ylim(graph_format_options['box_plot_ylim'])
    ax.set_xlabel('True Route Length', fontsize=graph_format_options['label_font_size'])
    ax.set_ylabel('Predicted Route Length', fontsize=graph_format_options['label_font_size'])
    # plt.title(reg_plot_title + f"\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.2f}")
    ax.set_title(f"\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.2f}", fontsize=graph_format_options['label_font_size'])
    ax.tick_params(axis='both', which='major', labelsize=graph_format_options['tick_font_size'])
    x_positions = ax.get_xticks()
    xticks_labels = ['unsolved' if int(x) == 0 else str(int(x)) for x in x_positions]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticks_labels)
    os.makedirs(os.path.dirname(reg_plot_path), exist_ok=True)
    plt.savefig(reg_plot_path)


# Confusion matrix plotter obtained from: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          filename=None,
                          fontsize=20, 
                          ): 
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    
    fontsize:      Font size for the text inside each square. Default is 12.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    plt.figure(figsize=figsize)
    vmax = graph_format_options["vmax"]
    cf_heatmap = sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories,annot_kws={"size": fontsize}, vmax=vmax)
    cf_heatmap.set_xticklabels(cf_heatmap.get_xticklabels(), fontsize=fontsize)
    cf_heatmap.set_yticklabels(cf_heatmap.get_yticklabels(), fontsize=fontsize)

    cbar_obj = cf_heatmap.collections[0].colorbar
    cbar_obj.ax.tick_params(labelsize=fontsize) 

    fig = cf_heatmap.get_figure()

    if xyplotlabels:
        plt.ylabel('True label', fontdict={'fontsize': fontsize}, labelpad=fontsize)  # Set font size for y-axis label
        plt.xlabel('Predicted label' + stats_text, fontdict={'fontsize': fontsize}, labelpad=fontsize)  # Set font size for x-axis label
    else:
        plt.xlabel(stats_text, fontdict={'fontsize': fontsize}, labelpad=fontsize)  # Set font size for x-axis label
    
    if title:
        plt.title(title, fontdict={'fontsize': fontsize * 1.5}, pad=fontsize)  # Set font size for title

    if filename is not None:
        fig.savefig(filename)
