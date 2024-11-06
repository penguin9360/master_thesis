from experiment_config import Experiment
from hyperopt import hp, fmin, tpe, Trials, rand
import chemprop
from chemprop.train import cross_validate, run_training
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from start_experiment import enable_regression, enable_multiclass, experiment_name, num_evals, NO_CUDA_OPTION, search_option
import numpy as np
import argparse
import sys

# pre-selected values
# epochs = 150
# depth = 6
# init_lr = 0.0001
# max_lr = 0.001
# batch_size = 64

# --batch-size Batch size (default 64)
# --message-hidden-dim <n> Hidden dimension of the messages in the MPNN (default 300)
# --depth <n> Number of message-passing steps (default 3)
# --dropout <n> Dropout probability in the MPNN & FFN layers (default 0)
# --activation <activation_type> The activation function used in the MPNN and FNN layers. Options include relu, leakyrelu, prelu, tanh, selu, and elu. (default relu)
# --epochs <n> How many epochs to train over (default 50)
# --warmup-epochs <n>: The number of epochs during which the learning rate is linearly incremented from init_lr to max_lr (default 2)
# --init-lr <n> Initial learning rate (default 0.0001)
# --max-lr <n> Maximum learning rate (default 0.001)
# --final-lr <n> Final learning rate (default 0.0001)

# grid search param sets for 1k and 10k
# epochs_set = [50, 100, 150, 200, 250]
# depth_set = [3, 5, 6, 7, 9]
# init_lr_set = [0.00005, 0.000075, 0.0001, 0.000125, 0.00015]
# max_lr_set = [0.0005, 0.00075, 0.001, 0.00125, 0.0015]
# batch_size_set = [32, 48, 64, 80, 96]

# reduced grid search space for 50k
epochs_set = [50, 200, 250]
depth_set = [3, 7, 9]
init_lr_set = [0.00005, 0.000075, 0.0001]
max_lr_set = [0.00075, 0.001, 0.00125]
batch_size_set = [48, 64, 96]

# boundaries for random search
lower_epochs = 50
upper_epochs = 250
lower_depth = 3
upper_depth = 9
lower_init_lr = 0.00005
upper_init_lr = 0.00015
lower_max_lr = 0.00075
upper_max_lr = 0.00125
lower_batch_size = 48
upper_batch_size = 96

hyper_params = {}
if search_option == "grid":
    hyper_params = {
        "epochs": hp.choice("epochs", epochs_set),
        "depth": hp.choice("depth", depth_set),
        "init_lr": hp.choice("init_lr", init_lr_set),
        "max_lr": hp.choice("max_lr", max_lr_set),
        "batch_size": hp.choice("batch_size", batch_size_set),
    }
elif search_option == "random":
    hyper_params = {
        'epochs': hp.quniform('epochs', lower_epochs, upper_epochs, 1),
        'depth': hp.quniform('depth', lower_depth, upper_depth, 1),
        'init_lr': hp.loguniform('init_lr', np.log(lower_init_lr), np.log(upper_init_lr)),
        'max_lr': hp.loguniform('max_lr', np.log(lower_max_lr), np.log(upper_max_lr)),
        'batch_size': hp.qloguniform('batch_size', np.log(lower_batch_size), np.log(upper_batch_size), 1)
    }
else:
    raise ValueError("Invalid search option")

experiment_mode = ""
training_set = ""
test_set = ""
validation_set = ""
chemprop_model_dir = ""
NO_CUDA_OPTION = False
hpo_folder = "hpo/"
hpo_logs = ""
hpo_models = ""
graph_format_options = {}


def set_experiment_params(experiment: Experiment, current_time):
    global experiment_name, experiment_mode, training_set, test_set, validation_set, chemprop_model_dir, NO_CUDA_OPTION, hpo_folder, hpo_logs, hpo_models, graph_format_options

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    training_set = experiment.training_set
    test_set = experiment.test_set
    validation_set = experiment.validation_set
    chemprop_model_dir = experiment.chemprop_model_dir
    NO_CUDA_OPTION = experiment.NO_CUDA_OPTION

    hpo_folder = "hpo/" + experiment_name + "/" + experiment_mode + "/"
    hpo_logs = hpo_folder + "logs/"
    hpo_models = hpo_folder + "models/" + current_time + "/"
    if not os.path.exists(hpo_logs):
        os.makedirs(hpo_logs)
    if not os.path.exists(hpo_models):
        os.makedirs(hpo_models)


def objective(params):
    training_arguments = [
        '--epochs', str(int(params['epochs'])),
        '--depth', str(int(params['depth'])),
        '--init_lr', str(params['init_lr']),
        '--max_lr', str(params['max_lr']),
        '--batch_size', str(int(params['batch_size'])),
        '--save_dir',  hpo_models,
        '--data_path', training_set,
        '--separate_val_path', validation_set,
        '--dataset_type', experiment_mode,
    ]
    if NO_CUDA_OPTION:
        training_arguments.append('--no_cuda')

    if experiment_mode == 'multiclass':
        training_arguments += ['--multiclass_num_classes', '4']

    args = chemprop.args.TrainArgs().parse_args(training_arguments)
    mean_score, std_score = cross_validate(args=args, train_func=run_training)
    print("Mean score:", mean_score, " | std score:", std_score)
    return mean_score


def run_hpo(num_evals, best_param_log_name):
    print(f"================================================== Starting GNN HPO experiment {experiment_name} {experiment_mode}, num_evals = {num_evals}, search_option = {search_option}... ==================================================")
    trials = Trials()
    if search_option == "grid":
        best = fmin(
            fn=objective,
            space=hyper_params,
            algo=tpe.suggest,
            max_evals=num_evals,
            trials=trials
        )

        best_hyperparams = {
        "epochs": epochs_set[best['epochs']],
        "depth": depth_set[best['depth']],
        "init_lr": init_lr_set[best['init_lr']],
        "max_lr": max_lr_set[best['max_lr']],
        "batch_size": batch_size_set[best['batch_size']],
        }

    elif search_option == "random":
        best = fmin(
            fn=objective,
            space=hyper_params,
            algo=rand.suggest, 
            max_evals=num_evals,
            trials=trials
        )

        best_hyperparams = {
            "epochs": int(best['epochs']),
            "depth": int(best['depth']),
            "init_lr": best['init_lr'],
            "max_lr": best['max_lr'],
            "batch_size": int(best['batch_size']),
        }
    else:
        raise ValueError("Invalid search option")
    
    
    print("Best hyperparameters:", best_hyperparams)
    
    finish_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    with open(best_param_log_name, 'a') as f:
        f.write(f"finished at {finish_time}\n\n")
        for key, value in best_hyperparams.items():
            f.write(f"{key}: {value}\n")
    print(f"================================================== Finished GNN HPO experiment {experiment_name} {experiment_mode}, search_option = {search_option}, num_evals = {num_evals}... ==================================================")


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Run parameter evaluation.')
    parser.add_argument('--task', type=str, choices=['regression', 'multiclass', 'both'], default='both',
                        help='Specify which task to run: regression, multiclass, or both.')
    args = parser.parse_args()

    # Set enable_regression and enable_multiclass based on the argument
    if args.task == 'regression':
        enable_regression = True
        enable_multiclass = False
    elif args.task == 'multiclass':
        enable_regression = False
        enable_multiclass = True
    elif args.task == 'both':
        enable_regression = True
        enable_multiclass = True
    else:
        print("Invalid task specified.")
        sys.exit(1)
    experiment_regression = Experiment(experiment_name, 'regression', NO_CUDA_OPTION, False, "", graph_format_options)
    experiment_multiclass = Experiment(experiment_name, 'multiclass', NO_CUDA_OPTION, False, "", graph_format_options)
    
    if enable_regression:
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        set_experiment_params(experiment_regression, current_time)
        best_param_log_name = hpo_logs + experiment_name + "_" + experiment_mode + "_best_params_" + current_time + ".txt"
        with open(best_param_log_name, 'w') as f:
            f.write(f"experiment name: {experiment_name}, mode: {experiment_mode}, num_evals: {num_evals}, search_option: {search_option}, \n started at {current_time}\n\n")
        run_hpo(num_evals, best_param_log_name)

    if enable_multiclass:
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        set_experiment_params(experiment_multiclass, current_time)
        best_param_log_name = hpo_logs + experiment_name + "_" + experiment_mode + "_best_params_" + current_time + ".txt"
        with open(best_param_log_name, 'w') as f:
            f.write(f"experiment name: {experiment_name}, mode: {experiment_mode}, num_evals: {num_evals}, search_option: {search_option},  \n started at {current_time}\n\n")
        run_hpo(num_evals, best_param_log_name)


    

    
