from gnn import run_gnn, set_parameters as set_gnn_parameters
from xg_boost import run_xgboost, set_parameters as set_xgboost_parameters
from gnn_result_analysis import run_gnn_result_analysis, set_parameters as set_gnn_analysis_parameters
from xg_boost_result_analysis import run_xg_boost_result_analysis, set_parameters as set_xg_boost_analysis_parameters
from helpers import write_hpo_slurm_file, set_parameters as set_helper_parameters
from experiment_config import Experiment

# experiment parameters - run_experiment must be set to True to run any experiments. Same for run_analysis. 
run_experiment = False
run_analysis = True
NO_CUDA_OPTION = False

# Note that currently only GNN HPO is supported. 
# If this flag is set to true, experiments and result analysis will not run, but the hpo.slurm file will be updated.
slurm_hpo_option = True
num_evals = 1
hpo_slurm_file = "hpo.slurm"

# these flags are to choose which evaluation sets will be used for inference
inference_option = False
inference_name = "50k" # '1k', '10k', '50k', '200k'

# enable/disable models
enable_gnn = True
enable_xgboost = False
enable_regression = True
enable_multiclass = True

experiment_name = "1k" # '1k', '10k', '50k'
cleanup = False
cleanup_name = "1k" # 'All' or '1k', '10k', '50k'

# make sure to toggle this flag to retrain with optimal params
retrain_gnn_with_optimal_param = False

epochs = 150
depth = 6
# To pass params, gnn uses list, xgboost uses dict
xgboost_model_param = {
    "n_estimators": epochs, 
    "max_depth": depth, 
    # "learning_rate": 0.1, 
    # "objective": 
    # "reg:squarederror"
    }

gnn_model_param = [
    "--epochs", str(epochs),
    "--depth", str(depth),
    # "hidden_size": 300, 
    # "dropout": 0.0, 
    # "batch_size": 32, 
    # "num_workers": 0
]

# params from HPO
# gnn_optimal_param_multiclass = [ # based on 1k HPO
#     "--epochs", "50",
#     "--depth", "5",
#     "--init_lr", "0.00005",
#     "--max_lr", "0.001",
#     "--batch_size", "96",
# ]

gnn_optimal_param_multiclass = [ # based on 10k HPO
    "--epochs", "250",
    "--depth", "7",
    "--init_lr", "0.00005",
    "--max_lr", "0.00075",
    "--batch_size", "64",
]

# gnn_optimal_param_regression = [ # based on 1k HPO
#     "--epochs", "150",
#     "--depth", "7",
#     "--init_lr", "0.000075",
#     "--max_lr", "0.0015",
#     "--batch_size", "48",
# ]

gnn_optimal_param_regression = [ # based on 10k HPO
    "--epochs", "200",
    "--depth", "3",
    "--init_lr", "0.0001",
    "--max_lr", "0.00125",
    "--batch_size", "96",
]


if __name__ == "__main__":
    experiment_regression = Experiment(experiment_name, 'regression', NO_CUDA_OPTION, inference_option, inference_name)
    experiment_multiclass = Experiment(experiment_name, 'multiclass', NO_CUDA_OPTION, inference_option, inference_name)

    if cleanup:
        # doesn't matter which instance we use, as the parameters are the same
        experiment_multiclass.cleanup(cleanup_name)
    
    if slurm_hpo_option:
        if enable_regression:
            write_hpo_slurm_file(hpo_slurm_file=hpo_slurm_file, experiment_name=experiment_name, experiment_mode='regression')
        if enable_multiclass:
            write_hpo_slurm_file(hpo_slurm_file=hpo_slurm_file, experiment_name=experiment_name, experiment_mode='multiclass')

    # experiment 
    if run_experiment and not slurm_hpo_option:
        if enable_gnn:
            if enable_regression:
                if retrain_gnn_with_optimal_param:
                    gnn_model_param = gnn_optimal_param_regression
                set_gnn_parameters(experiment_regression, gnn_model_param)
                set_helper_parameters(experiment_regression)
                run_gnn()
            if enable_multiclass:
                if retrain_gnn_with_optimal_param:
                    gnn_model_param = gnn_optimal_param_multiclass
                set_gnn_parameters(experiment_multiclass, gnn_model_param)
                set_helper_parameters(experiment_multiclass)
                run_gnn()
        
        if enable_xgboost:
            if enable_regression:
                set_xgboost_parameters(experiment_regression, xgboost_model_param)
                set_helper_parameters(experiment_regression)
                run_xgboost()
            if enable_multiclass:
                set_xgboost_parameters(experiment_multiclass, xgboost_model_param)
                set_helper_parameters(experiment_multiclass)
                run_xgboost()

    # analysis
    if run_analysis and not slurm_hpo_option:
        if enable_gnn:
            if enable_regression:
                set_gnn_analysis_parameters(experiment_regression)
                set_helper_parameters(experiment_regression)
                run_gnn_result_analysis()
            if enable_multiclass:
                set_gnn_analysis_parameters(experiment_multiclass)
                set_helper_parameters(experiment_multiclass)
                run_gnn_result_analysis()

        if enable_xgboost:
            if enable_regression:
                set_xg_boost_analysis_parameters(experiment_regression)
                set_helper_parameters(experiment_regression)
                run_xg_boost_result_analysis()
            if enable_multiclass:
                set_xg_boost_analysis_parameters(experiment_multiclass)
                set_helper_parameters(experiment_multiclass)
                run_xg_boost_result_analysis()


    
    