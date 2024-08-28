from gnn import run_gnn, set_parameters as set_gnn_parameters
from xg_boost import run_xgboost, set_parameters as set_xgboost_parameters
from gnn_result_analysis import run_gnn_result_analysis, set_parameters as set_gnn_analysis_parameters
from xg_boost_result_analysis import run_xg_boost_result_analysis, set_parameters as set_xg_boost_analysis_parameters

from experiment_config import Experiment

# experiment parameters - run_experiment must be set to True to run any experiments. Same for run_analysis. 
run_experiment = True
run_analysis = True
NO_CUDA_OPTION = False

inference_option = False
inference_name = "50ktest2" # '1k', '10k', '50k' or '50ktest2', '200k'

# enable/disable models
enable_gnn = True
enable_xgboost = True
enable_regression = True
enable_multiclass = True

experiment_name = "1k" # '1k', '10k', '50k', or '50ktest2', '200k'
cleanup = True
cleanup_name = "1k" # 'All' or '1k', '10k', '50k', or '50ktest2', '200k'

if __name__ == "__main__":
    experiment_regression = Experiment(experiment_name, 'regression', NO_CUDA_OPTION, inference_option, inference_name)
    experiment_multiclass = Experiment(experiment_name, 'multiclass', NO_CUDA_OPTION, inference_option, inference_name)

    if cleanup:
        # doesn't matter which instance we use, as the parameters are the same
        experiment_multiclass.cleanup(cleanup_name)

    # experiment 
    if run_experiment:
        if enable_gnn:
            if enable_regression:
                set_gnn_parameters(experiment_regression)
                run_gnn()
            if enable_multiclass:
                set_gnn_parameters(experiment_multiclass)
                run_gnn()
        
        if enable_xgboost:
            if enable_regression:
                set_xgboost_parameters(experiment_regression)
                run_xgboost()
            if enable_multiclass:
                set_xgboost_parameters(experiment_multiclass)
                run_xgboost()

    # analysis
    if run_analysis:
        if enable_gnn:
            if enable_regression:
                set_gnn_analysis_parameters(experiment_regression)
                run_gnn_result_analysis()
            if enable_multiclass:
                set_gnn_analysis_parameters(experiment_multiclass)
                run_gnn_result_analysis()

        if enable_xgboost:
            if enable_regression:
                set_xg_boost_analysis_parameters(experiment_regression)
                run_xg_boost_result_analysis()
            if enable_multiclass:
                set_xg_boost_analysis_parameters(experiment_multiclass)
                run_xg_boost_result_analysis()


    
    