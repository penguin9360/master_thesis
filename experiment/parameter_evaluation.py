from experiment_config import Experiment
from hyperopt import hp, fmin, tpe, Trials
import chemprop
from chemprop.train import cross_validate, run_training
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# experiment parameters - run_experiment must be set to True to run any experiments. Same for run_analysis. 
NO_CUDA_OPTION = False

enable_regression = True
enable_multiclass = True

experiment_name = "1k" # '1k', '10k', '50k'
num_evals = 50


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

epochs_set = [50, 100, 150, 200, 250]
depth_set = [3, 5, 6, 7, 9]
init_lr_set = [0.00005, 0.000075, 0.0001, 0.000125, 0.00015]
max_lr_set = [0.0005, 0.00075, 0.001, 0.00125, 0.0015]
batch_size_set = [32, 48, 64, 80, 96]


hyper_params = {
    "epochs": hp.choice("epochs", epochs_set),
    "depth": hp.choice("depth", depth_set),
    "init_lr": hp.choice("init_lr", init_lr_set),
    "max_lr": hp.choice("max_lr", max_lr_set),
    "batch_size": hp.choice("batch_size", batch_size_set),
}

experiment_mode = ""
training_set = ""
test_set = ""
validation_set = ""
chemprop_model_dir = ""
NO_CUDA_OPTION = False
hpo_folder = "hpo/"


def set_experiment_params(experiment: Experiment):
    global experiment_name, experiment_mode, training_set, test_set, validation_set, chemprop_model_dir, NO_CUDA_OPTION

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    training_set = experiment.training_set
    test_set = experiment.test_set
    validation_set = experiment.validation_set
    chemprop_model_dir = experiment.chemprop_model_dir
    NO_CUDA_OPTION = experiment.NO_CUDA_OPTION


def objective(params):
    training_arguments = [
        '--epochs', str(params['epochs']),
        '--depth', str(params['depth']),
        '--init_lr', str(params['init_lr']),
        '--max_lr', str(params['max_lr']),
        '--batch_size', str(params['batch_size']),
        '--save_dir', chemprop_model_dir + "_hpo",
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
    print(f"================================================== Starting GNN HPO experiment {experiment_name} {experiment_mode}, num_evals = {num_evals}... ==================================================")
    trials = Trials()
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
    print("Best hyperparameters:", best_hyperparams)
    
    finish_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    with open(best_param_log_name, 'a') as f:
        f.write(f"finished at {finish_time}\n\n")
        for key, value in best_hyperparams.items():
            f.write(f"{key}: {value}\n")
    print(f"================================================== Finished GNN HPO experiment {experiment_name} {experiment_mode}... ==================================================")


if __name__ == "__main__":
    if not os.path.exists(hpo_folder):
        os.makedirs(hpo_folder)
    experiment_regression = Experiment(experiment_name, 'regression', NO_CUDA_OPTION, False, "")
    experiment_multiclass = Experiment(experiment_name, 'multiclass', NO_CUDA_OPTION, False, "")
    
    if enable_regression:
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        set_experiment_params(experiment_regression)
        best_param_log_name = hpo_folder + experiment_name + "_" + experiment_mode + "_best_params_" + current_time + ".txt"
        with open(best_param_log_name, 'w') as f:
            f.write(f"experiment name: {experiment_name}, mode: {experiment_mode}, num_evals: {num_evals} \n started at {current_time}\n\n")
        run_hpo(num_evals, best_param_log_name)

    if enable_multiclass:
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        set_experiment_params(experiment_multiclass)
        best_param_log_name = hpo_folder + experiment_name + "_" + experiment_mode + "_best_params_" + current_time + ".txt"
        with open(best_param_log_name, 'w') as f:
            f.write(f"experiment name: {experiment_name}, mode: {experiment_mode}, num_evals: {num_evals} \n started at {current_time}\n\n")
        run_hpo(num_evals, best_param_log_name)


    

    
