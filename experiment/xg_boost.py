import os
import pandas as pd
import numpy as np
import csv
# import h5py
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
# from rdkit.Chem import MACCSkeys
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import auc, accuracy_score, recall_score
# from sklearn.metrics import roc_curve, roc_auc_score
from xgboost_distribution import XGBDistribution
from xgboost import XGBClassifier, XGBRegressor, DMatrix

from collections import Counter
from helpers import train_test_split, get_files_in_directory, is_empty_folder
from experiment_config import Experiment
# from xg_boost_result_analysis import plot_residuals
# from confusion_matrix_plotter import make_confusion_matrix
import threading

# NEXT - get 1000 data and split (70% training, 15% validation, 15% test) √
#      - set new AIZ search limit on route length to ? (refer to Mike's 2018 paper) right now default is 6 √
#      - New pipeline of GNN (Chemprop) and xgboost 
#        - xgboost with ECFP encoding (fingerprint size = 2048, radius = 3) √
#        - GNN with chemprop 
#        - Uncertainty estimation for both models
#        - route length (number_of_step) as multi-class classification (label unsolvable as -1) √
#      - Larger data sets
#      - Reevaluate the pipeline
#      - Get performance value for each class (F1 and MCC...F1 and MCC for overall performance) 
#      - Visualization of results:
#        - Bar plot per class (https://towardsdatascience.com/a-different-way-to-visualize-classification-results-c4d45a0a37bb)
#        - confusion matrix √
#      - 

# Declare missing variables and set them to empty strings
experiment_name = ""
experiment_mode = ""
results_dir = ""
figures_dir = ""
UNSOLVED_LENGTH = 0
combined_set = ""
training_set = ""
test_set = ""
extract_file_from_hdf = ""
xgboost_prediction = ""
xgboost_model_dir = ""

def set_parameters(experiment: Experiment):
    global experiment_name, experiment_mode, results_dir, figures_dir, UNSOLVED_LENGTH, combined_set, training_set, test_set, extract_file_from_hdf, xgboost_prediction, xgboost_model_dir

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    results_dir = experiment.results_dir
    figures_dir = experiment.figures_dir
    UNSOLVED_LENGTH = experiment.UNSOLVED_LENGTH
    combined_set = experiment.combined_set
    training_set = experiment.training_set
    test_set = experiment.test_set
    extract_file_from_hdf = experiment.extract_file_from_hdf
    xgboost_prediction = experiment.xgboost_prediction
    xgboost_model_dir = experiment.xgboost_model_dir


def run_xgboost():
    # ================================================================== PROGRAM STARTS HERE ===================================================================
    print(f"================================================== Starting XGBoost experiment {experiment_name} {experiment_mode}... ==================================================")

    file_list = get_files_in_directory(results_dir)
    raw_smiles = []
    route_lengths = []
    unsolved_counter = 0

    # Read all HDF files and extract info
    if os.path.isfile(extract_file_from_hdf):
        print(f"Reading from HDF, extraction file exists. Loading from file {extract_file_from_hdf}...")
        with open(extract_file_from_hdf, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            combined_data = list(reader)
            raw_smiles, route_lengths = zip(*combined_data)
        for i in range(len(route_lengths)):
            if int(route_lengths[i]) == 0:
                unsolved_counter += 1
    else:
        print(f"Reading from HDF and writing to extraction file {extract_file_from_hdf}...")
        for f in file_list:
            data = pd.read_hdf(f, "table")
            for i in range(len(data)):
                smile = data['target'][i]
                is_solved = data['is_solved'][i]
                route_length = data['number_of_steps'][i]
                if not is_solved:
                    unsolved_counter += 1
                    print(f"Unsolved molecule #{unsolved_counter}: {smile}, with original route length {route_length} set to {UNSOLVED_LENGTH}")
                    route_length = UNSOLVED_LENGTH
                raw_smiles.append(smile)
                route_lengths.append(route_length)
                combined_data = list(zip(raw_smiles, route_lengths))

        with open(extract_file_from_hdf, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smile', 'route_length'])
            writer.writerows(combined_data)

    print(f"       =================================================================================================================================================== \n\
        Import completed. {len(raw_smiles) - unsolved_counter} out of {len(raw_smiles)} were solved. Starting encoding... \n \
        ===================================================================================================================================================")

    # regroup route lengths into 4 classes --
    # 1. 1-2 steps
    # 2. 3-5 steps
    # 3. 6+ steps
    # 0. unsolved (length == 0 as specified earlier)

    route_lengths_regrouped = []
    # explain by how easy reactions take place
    for i in route_lengths:
        if int(i) == 1 or int(i) == 2:
            route_lengths_regrouped.append("1")
        if int(i) >= 3 and int(i) <= 5:
            route_lengths_regrouped.append("2")
        if int(i) >= 6:
            route_lengths_regrouped.append("3")
        if int(i) == 0:
            route_lengths_regrouped.append("0")

    print(f"Regrouped route lengths:\n {Counter(route_lengths_regrouped).keys()} \n {Counter(route_lengths_regrouped).values()}")


    # to do multiclass, change route_lengths to route_lengths_regrouped
    params = {
        'training_set': training_set,
        'test_set': test_set,
        'combined_set': combined_set,
        'experiment_mode': experiment_mode
    }
    X_train, X_test, y_train, y_test = train_test_split(raw_smiles, route_lengths, test_size=0.2, y_regrouped=route_lengths_regrouped, params=params)

    print(f"Performing {experiment_mode}. Train_test_split:\n X_train: ", len(X_train), ", X_test: ", len(X_test), ", y_train: ", len(y_train), ", y_test: ", len(y_test))


    # ECFP encoding
    feature_list = []
    feature_list_X_train = []
    feature_list_X_test = []
    """
        Inputs:
        
        - smiles ... SMILES string of input compound
        - R ... maximum radius of circular substructures
        - L ... fingerprint-length
        - use_features ... if false then use standard DAYLIGHT atom features, if true then use pharmacophoric atom features
        - use_chirality ... if true then append tetrahedral chirality flags to atom features
        
        Outputs:
        - np.array(feature_list) ... ECFP with length L and maximum radius R
        """
    # Default Values from https://www.blopig.com/blog/2022/11/how-to-turn-a-smiles-string-into-an-extended-connectivity-fingerprint-using-rdkit/ 
    R = 3
    L = 2048
    use_features = True
    use_chirality = True

    # for i in range(len(raw_smiles)):
    #     molecule_encoded = AllChem.MolFromSmiles(raw_smiles[i])
    #     feature_list.append(np.array(AllChem.GetMorganFingerprintAsBitVect(molecule_encoded, radius = R, nBits = L, useFeatures = use_features, useChirality = use_chirality)))

    for i in range(len(X_train)):
        molecule_encoded = AllChem.MolFromSmiles(X_train[i])
        feature_list_X_train.append(np.array(AllChem.GetMorganFingerprintAsBitVect(molecule_encoded, radius = R, nBits = L, useFeatures = use_features, useChirality = use_chirality)))

    for i in range(len(X_test)):
        molecule_encoded = AllChem.MolFromSmiles(X_test[i])
        feature_list_X_test.append(np.array(AllChem.GetMorganFingerprintAsBitVect(molecule_encoded, radius = R, nBits = L, useFeatures = use_features, useChirality = use_chirality)))


    # print("Molecules encoded. Size of feature list: ", len(feature_list), len(feature_list[0]), feature_list[:5])
    print(f"Molecules encoded. X_train feature size: {np.array(feature_list_X_train).shape}; X_test feature size: {np.array(feature_list_X_test).shape}. Start fitting...")

    # # fit with XGBoost Classifier 
    # X, y = np.array(feature_list), np.array(route_lengths_regrouped)
    X_train = np.array(feature_list_X_train)
    X_test_copy = np.array(X_test)
    X_test = np.array(feature_list_X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # @TODO: This may need to be fixed: https://towardsdatascience.com/straightforward-stratification-bb0dcfcaf9ef 
    # also: https://github.com/aiqc/aiqc 
    #@TODO: Add saved train/test to xgboost

    # This is the old working way - now testing the new custom-defined train_test_split below
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)




    # model = XGBDistribution(
    #     distribution="normal",
    #     n_estimators=500,
    #     early_stopping_rounds=10
    # )

    # LabelEncoder magic - otherwise it throws ValueError: Invalid classes inferred from unique values of `y`
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    if experiment_mode == 'multiclass':

        if os.path.exists(xgboost_model_dir):
            print(f"Model directory exists. Loading model from {xgboost_model_dir}...")
            model = XGBClassifier()
            model.load_model(xgboost_model_dir)
        else:
            model = XGBClassifier()
            # model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            model.fit(X_train, y_train, verbose=2)
            # model.fit(np.reshape(X_train, (-1, 1)), np.reshape(y_train, (-1, 1)), verbose=2)
            # model.fit(DMatrix(np.reshape(X_train, (-1, 1))), DMatrix(np.reshape(y_train, (-1, 1))), verbose=2)
            if not os.path.exists("xgboost/model"):
                os.makedirs("xgboost/model")
            model.save_model(xgboost_model_dir)
            
            print(f"Fitting finished. Model saved to {xgboost_model_dir}. Calculating metrics in a different thread...")

        # cross validataion - this may lead to a segfault on large datasets for some reasons (https://github.com/dmlc/xgboost/issues/9369), so using multi-threading 
        def calculate_metrics(model, X_train, y_train):
            scores = cross_val_score(model, X_train, y_train, cv=5)
            print("Mean cross-validation score: %.2f" % scores.mean())

            kfold = KFold(n_splits=10, shuffle=True)
            kf_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold )
            print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

        calc_metric_thread = threading.Thread(target=calculate_metrics, name="metrics", args=[model, X_train, y_train])
        calc_metric_thread.start()

        preds = model.predict(X_test)
        # print("Checking if metric calculation thread is still alive: ", calc_metric_thread.is_alive())
        # Create a DataFrame with the predictions and corresponding SMILES
        predictions_df = pd.DataFrame({'smiles': X_test_copy, 'route_length': preds})

        # Save the DataFrame to a CSV file
        if not os.path.exists("xgboost/data"):
            os.makedirs("xgboost/data")   
        predictions_df.to_csv(xgboost_prediction, index=False)

        # inverse transfrom from the LabelEncoder magic
        preds = le.inverse_transform(preds)

        print("Classification Report: \n", metrics.classification_report(y_test, preds))

        categories = ['unsolved', '1-2', '3-5', '6+']
        confusion_matrix_test = metrics.confusion_matrix(y_test, preds)
        print("Confusion Matrix on y_test: \n", confusion_matrix_test)
        # confusion_matrix_train = metrics.confusion_matrix(y_train, preds)
        # print("Confusion Matrix on y_train: \n", confusion_matrix_train)

        # start plotting 
        print("Plotting Confusion Matrix graph...")
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        plot_file_prefix = figures_dir + experiment_name + "_xgboost_" + experiment_mode
        plot_title_prefix = experiment_name + "_xgboost_" + experiment_mode
        # make_confusion_matrix(confusion_matrix_test, categories=categories, figsize=(15, 15), filename=plot_file_prefix + "_confusion_matrix.png", title=plot_title_prefix + "_confusion_matrix")
    
        # As discussed with Alan, mosaic graph is no longer necessary
        # print("Plotting mosaic graph...")
        # nclass_classification_mosaic_plot(n_classes=len(confusion_matrix_test.tolist()), results=confusion_matrix_test.tolist(), filename=plot_file_prefix + "_mosaic.png", title=plot_title_prefix + "_mosaic")
        # nclass_classification_mosaic_plot(n_classes=len(confusion_matrix_train.tolist()), results=confusion_matrix_train.tolist(), filename=figures_dir + experiment_name + "_mosaic_train.png")

        if calc_metric_thread.is_alive():
            print("Main thread execution finished but metric calculation is not finished yet. Waiting for it to finish...")
            calc_metric_thread.join()


        # mean, std = preds.loc, preds.scale
        # print("Fitting complete.\n mean: ", mean, " std: ", std, "\nPlotting graph...")
        # plot_residuals(y_true=y_test, y_pred=preds.loc, y_err=preds.scale, filename=plot_file_prefix + "_residuals.png", title=plot_title_prefix + "_residuals")

        # # Load SMILES from a file
        # with open(SMILES, 'r') as f:
        #     smiles_list = f.readlines()

        # # Remove newline characters
        # smiles_list = [smiles.strip() for smiles in smiles_list]

        # # Generate MACCS fingerprints for each SMILES !!! forget about this !!! DO NOT USE MACCS
        # maccs_list = []
        # for smiles in smiles_list:
        #     mol = Chem.MolFromSmiles(smiles)
        #     maccs = MACCSkeys.GenMACCSKeys(mol)
        #     maccs_list.append(maccs)

        # # print(maccs_list[:10])

        # # Convert MACCS fingerprints to a binary array
        # # ctr = 0
        # maccs_array = []
        # for maccs in maccs_list:
        #     maccs_bits = [int(bit) for bit in maccs.ToBitString()]
        #     maccs_array.append(maccs_bits)
        #     # print(maccs_array[ctr])
        #     # ctr += 1

        # maccs_array = np.array(maccs_array)

        # # parse results from HPC cluster in .hdf format
        # # >>> data.columns
        # # Index(['index', 'target', 'search_time', 'first_solution_time',
        # #       'first_solution_iteration', 'number_of_nodes', 'max_transforms',
        # #       'max_children', 'number_of_routes', 'number_of_solved_routes',
        # #       'top_score', 'is_solved', 'number_of_steps', 'number_of_precursors',
        # #       'number_of_precursors_in_stock', 'precursors_in_stock',
        # #       'precursors_not_in_stock', 'precursors_availability',
        # #       'policy_used_counts', 'profiling', 'top_scores', 'trees'],
        # #      dtype='object')

        # data = pd.read_hdf("joined_result.hdf", "table")
        # for i in range(len(data)):
        #     print(data['target'][i], data['number_of_steps'][i])

        # # fitting ECFP encoding to route length 
        # X, y = maccs_array, data.number_of_steps
        # X_train, X_test, y_train, y_test = train_test_split(X, y)

        # model = XGBDistribution(
        #     distribution="normal",
        #     n_estimators=500,
        #     early_stopping_rounds=10
        # )
        # model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        # preds = model.predict(X_test)
        # mean, std = preds.loc, preds.scale
        # print("mean: ", mean, " std: ", std)
    else:
        if os.path.exists(xgboost_model_dir):
            print(f"Model directory exists. Loading model from {xgboost_model_dir}...")
            model = XGBRegressor()
            model.load_model(xgboost_model_dir)
        else:
            model = XGBRegressor()
            model.fit(X_train, y_train, verbose=2)
            if not os.path.exists("xgboost/model"):
                os.makedirs("xgboost/model")
            model.save_model(xgboost_model_dir)
            print(f"Fitting finished. Model saved to {xgboost_model_dir}. Calculating metrics...")
        preds = model.predict(X_test)
        predictions_df = pd.DataFrame({'smiles': X_test_copy, 'route_length': preds})
        if not os.path.exists("xgboost/data"):
            os.makedirs("xgboost/data")        
        predictions_df.to_csv(xgboost_prediction, index=False)

        print("Mean Squared Error: ", metrics.mean_squared_error(y_test, preds))
        print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, preds))
        print("R2 Score: ", metrics.r2_score(y_test, preds))
        print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, preds)))
    
    print(f"================================================== Finished XGBoost experiment {experiment_name} {experiment_mode}... ==================================================")

if __name__ == "__main__":
    run_xgboost()