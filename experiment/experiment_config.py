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
    validation_set = ""
    chemprop_prediction = ""
    chemprop_model_dir = ""
    xgboost_prediction = ""
    xgboost_model_dir = ""
    extract_file_from_hdf = ""
    inference_option = False
    inference_name = ""
    inference_combined_set = ""
    inference_test_set = ""

    UNSOLVED_LENGTH = 0
    NO_CUDA_OPTION = True

    cleanable_directories = ["gnn", "xgboost", "figures", "result_extract", "train_test"]


    def __init__(self, name, mode, no_cuda_option, inference_option, inference_name):
        self.experiment_name = name
        self.extract_file_from_hdf = "./result_extract/" + self.experiment_name + "_extract.csv"
        self.inference_option = inference_option
        self.inference_name = inference_name
        self.experiment_mode = mode
        self.NO_CUDA_OPTION = no_cuda_option
        
        self.results_dir = "results/" + self.experiment_name + "/"
        self.combined_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_combined.csv"
        self.training_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_" + self.experiment_mode + "_train.csv"
        self.validation_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_" + self.experiment_mode + "_validation.csv"
        self.xgboost_model_dir = "./xgboost/model/" + self.experiment_name + "_" + self.experiment_mode + ".json"
        self.chemprop_model_dir = "gnn/model/" + self.experiment_name + "_" + self.experiment_mode

        self.test_set = "./train_test/" + self.experiment_name + "/" + self.experiment_name + "_" + self.experiment_mode + "_test.csv"
        self.figures_dir = "figures/" + self.experiment_name + "/"
        self.chemprop_prediction = "gnn/data/" + self.experiment_name + "_" + self.experiment_mode + "_prediction"
        self.xgboost_prediction = "./xgboost/data/" + self.experiment_name + "_" + self.experiment_mode + "_prediction"

        if self.inference_option:
            self.test_set = "./train_test/" + self.experiment_name + "_inference_" + self.inference_name + "/" + self.experiment_name + "_inference_" + self.inference_name + "_" + self.experiment_mode + "_test.csv"
            self.inference_combined_set = "./train_test/" + self.inference_name + "/" + self.inference_name + "_combined.csv"
            self.inference_test_set = "./train_test/" + self.inference_name + "/" + self.inference_name + "_" + self.experiment_mode + "_test.csv"
            self.figures_dir = "figures/" + self.experiment_name + "_inference_" + self.inference_name + "/"
            self.chemprop_prediction = "gnn/data/" + self.experiment_name + "_inference_" + self.inference_name + "_" + self.experiment_mode + "_prediction"
            self.xgboost_prediction = "./xgboost/data/" + self.experiment_name+ "_inference_" + self.inference_name + "_" + self.experiment_mode + "_prediction"
            

        
    def cleanup(self, name):
        if name == "All":   
            for directory in self.cleanable_directories:
                shutil.rmtree(directory, ignore_errors=True)
                os.makedirs(directory)
                print(f"Directory {directory} cleaned.")
        else:
            for directory in self.cleanable_directories:
                for directory in self.cleanable_directories:
                    for root, dirs, files in os.walk(directory):
                        for f in files:
                            if (name + "_") in f:
                                file_path = os.path.join(root, f)
                                os.remove(file_path)
                                print(f"File {file_path} removed.")
                        for d in dirs:
                            if name in d:
                                if str(d).split(name)[-1] == '' or 'model' in root or 'figures' in root or 'train_test' in root:
                                    dir_path = os.path.join(root, d)
                                    shutil.rmtree(dir_path, ignore_errors=True)
                                    print(f"Directory {dir_path} removed.")
        print(f"=============================== Cleanup for {name} completed. Note that all models directory are not removed. Remove manually if needed. ===============================")
                