import os
import shutil


class Experiment:

    experiment_name = ""
    experiment_mode = ""

    results_dir = ""
    figures_dir = ""
    combined_set = ""
    training_set = ""
    test_set = ""
    chemprop_prediction = ""
    chemprop_model_dir = ""
    xgboost_prediction = ""
    xgboost_model_dir = ""
    extract_file_from_hdf = ""

    UNSOLVED_LENGTH = 0
    NO_CUDA_OPTION = True

    cleanable_directories = ["gnn", "xgboost", "figures", "result_extract", "train_test"]


    def __init__(self, name, mode):
        self.experiment_name = name
        self.experiment_mode = mode
        self.results_dir = "results/" + self.experiment_name + "/"
        self.figures_dir = "figures/" + self.experiment_name + "/"
        self.combined_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_combined.csv"
        self.training_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_" + self.experiment_mode + "_train.csv"
        self.test_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_" + self.experiment_mode + "_test.csv"
        self.chemprop_prediction = "gnn/data/" + self.experiment_name + "_" + self.experiment_mode + "_prediction"
        self.chemprop_model_dir = "gnn/model/" + self.experiment_name + "_" + self.experiment_mode
        self.xgboost_prediction = "./xgboost/data/" + self.experiment_name + "_" + self.experiment_mode + "_prediction"
        self.xgboost_model_dir = "./xgboost/model/" + self.experiment_name + "_" + self.experiment_mode + ".json"
        self.extract_file_from_hdf = "./result_extract/" + self.experiment_name + "_extract.csv"

    def cleanup(self):
        for directory in self.cleanable_directories:
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory)
