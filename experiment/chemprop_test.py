import chemprop
from helpers import get_files_in_directory, plot_tensorboard_learning_curves, set_parameters
import subprocess
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
from experiment_config import Experiment
import matplotlib as mpl
import matplotlib.pyplot as plt
from start_experiment import graph_format_options

# result_200k = get_files_in_directory("./results/200k")
# print(len(result_200k))

# experiment_mode = 'regression'
experiment_mode = 'multiclass'

experiment_regression = Experiment("1k", 'regression', False, False, "", graph_format_options)
experiment_multiclass = Experiment("1k", 'multiclass', False, False, "", graph_format_options)
if experiment_mode == 'regression':
    set_parameters(experiment_regression)
else:
    set_parameters(experiment_multiclass)
epochs = 50
gnn_optimal_param_multiclass = [ # based on 1k HPO
    "--epochs", str(epochs), #optimal 50
    "--depth", "5",
    "--init_lr", "0.00005",
    "--max_lr", "0.001",
    "--batch_size", "96",
]

gnn_optimal_param_regression = [ # based on 1k HPO
    "--epochs", str(epochs), # optimal 150
    "--depth", "7",
    "--init_lr", "0.000075",
    "--max_lr", "0.0015",
    "--batch_size", "48",
]

training_set = "./train_test/1k/1k_" + experiment_mode + "_train.csv"
test_set = "./train_test/1k/1k_" + experiment_mode + "_test.csv"
val_set = "./train_test/1k/1k_" + experiment_mode + "_validation.csv"
chemprop_model_dir = "./gnn/model/1k_" + experiment_mode
training_arguments = [                                  
                '--data_path', training_set,
                '--dataset_type', experiment_mode,
                '--save_dir', chemprop_model_dir,
            ]
if experiment_mode == 'multiclass':
    training_arguments.append('--multiclass_num_classes')
    training_arguments.append("4")
training_arguments_2 = ["--separate_val_path", val_set, "--separate_test_path", test_set]
training_arguments+=(training_arguments_2)

# training_arguments.append("--split_sizes")
# training_arguments.extend(map(str, (0.999, 0.001, 0.0)))
if experiment_mode == 'regression':
    training_arguments += gnn_optimal_param_regression
if experiment_mode == 'multiclass':
    training_arguments += gnn_optimal_param_multiclass
training_arguments.extend([
    '--metric', 'accuracy',
    '--extra_metrics', 'f1',
])

args = chemprop.args.TrainArgs().parse_args(training_arguments)

mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
print("Chemprop training score: \nMean: ", mean_score, " | std: ", std_score)

tensorboard_log_dir = "./gnn/model/1k_" + experiment_mode + "/fold_0/model_0"


#https://www.kaggle.com/code/aleksandrkruchinin/tensorboard-events-to-matplotlib

#below code is borrowed from
#https://gist.github.com/tomrunia/1e1d383fb21841e8f144

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        # 'compressedHistograms': 10,
        # 'images': 0,
        # 'scalars': 100,
        # 'histograms': 1
    }
    

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())
    print(f"Event keys {experiment_mode}: ", event_acc.Tags())
    # print("Event Scalars: ", event_acc.Scalars('train_loss'))

    training_accuracies = event_acc.Scalars('train_loss')
    validation_accuracies = event_acc.Scalars('validation_rmse')
    steps = min(len(training_accuracies), len(validation_accuracies))
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] 
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    # plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.savefig('./figures/gnn_loss_test.png')


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


res = {}
for f in get_files_in_directory(tensorboard_log_dir):
    res[f] = {}
    event_acc = EventAccumulator(f, size_guidance={"scalars": 0})
    event_acc.Reload()

    for scalar in event_acc.Tags()["scalars"]:
        res[f].update(parse_tensorboard(f, [scalar]))

res_df = pd.DataFrame(res)

display_cap = 3

# useful, commented out for now

# for key in res_df.keys():
#     print(key)
#     for key2 in res[key].keys():
#         print("\n", key2)
#         print(res[key][key2][:display_cap])
#         if 'test' not in key2:
#             print("...")
#     print("=====================================================================================================")

# print(res['./gnn/model/1k_multiclass/fold_0/model_0/events.out.tfevents.1725394672.nodelogin01']['validation_cross_entropy'][res['./gnn/model/1k_multiclass/fold_0/model_0/events.out.tfevents.1725394672.nodelogin01']['validation_cross_entropy']['step'] <= 2000])


for f in get_files_in_directory(tensorboard_log_dir):
    if '.pt' in f:
        continue
    print(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=f, type=str, help='logdir to event file')
    parser.add_argument('--smooth', default=25, type=float, help='window size for average smoothing')
    parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    plot_tensorboard_learning_curves(params, epochs)