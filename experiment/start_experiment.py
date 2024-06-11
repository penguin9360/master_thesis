from gnn import run_gnn, set_parameters as set_gnn_parameters
from xg_boost import run_xgboost, set_parameters as set_xgboost_parameters

from experiment_config import Experiment

experiment_name = "50ktest2" # '1k' or '50ktest2'
cleanup = False

if __name__ == "__main__":
    experiment_regression = Experiment(experiment_name, 'regression')
    experiment_multiclass = Experiment(experiment_name, 'multiclass')

    if cleanup:
        # doesn't matter which instance we use, as the parameters are the same
        experiment_multiclass.cleanup()

    set_gnn_parameters(experiment_regression)
    set_xgboost_parameters(experiment_regression)

    run_gnn()
    run_xgboost()

    set_gnn_parameters(experiment_multiclass)
    set_xgboost_parameters(experiment_multiclass)
    
    run_gnn()
    run_xgboost()

    # from gnn_result_analysis import set_parameters as set_gnn_analysis_parameters
    # set_gnn_analysis_parameters(experiment)
