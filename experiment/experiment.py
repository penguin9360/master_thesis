import os
import pandas as pd
import numpy as np
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
from xgboost import XGBClassifier
from plotter import make_confusion_matrix, nclass_classification_mosaic_plot
from collections import Counter
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

experiment_name = "50ktest2"
# experiment_name = "1k"
results_dir = "results/" + experiment_name + "/"
figures_dir = "figures/" + experiment_name + "/"
UNSOLVED_LENGTH = 0

def get_files_in_directory(directory):
    file_list = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    
    return file_list


def plot_residuals(y_true, y_pred, y_err):
    fig, ax = plt.subplots()
    ax.errorbar(
        y_true,
        y_true - y_pred,
        yerr=y_err,
        marker="o",
        linestyle="None",
        c="k",
        markersize=2.5,
        linewidth=0.5,
    )
    ax.axhline(0, c="k", linestyle="--")
    ax.set_xlabel("y_test")
    ax.set_ylabel("y_test - y_pred")
    plt.savefig(figures_dir + experiment_name + "_residual.png")



# ================================================================== PROGRAM STARTS HERE ===================================================================

file_list = get_files_in_directory(results_dir)
raw_smiles = []
route_lengths = []
unsolved_counter = 0

# Read all HDF files and extract info
for f in file_list:
    data = pd.read_hdf(f, "table")
    for i in range(len(data)):
        smile = data['target'][i]
        is_solved = data['is_solved'][i]
        route_length = data['number_of_steps'][i]
        # print(smile, is_solved, route_length)
        if not is_solved:
            unsolved_counter += 1
            print(f"Unsolved molecule #{unsolved_counter}: {smile}, with original route length {route_length} set to {UNSOLVED_LENGTH}")
            route_length = UNSOLVED_LENGTH
        raw_smiles.append(smile)
        route_lengths.append(route_length)

print(f"       =================================================================================================================================================== \n\
      Import completed. {len(raw_smiles) - unsolved_counter} out of {len(raw_smiles)} were solved. Starting encoding... \n \
      ===================================================================================================================================================")

# regroup route lengths into 4 classes --
# 1. 1-2 steps
# 2. 3-5 steps
# 3. 6+ steps
# 4. unsolved (length == 0 as specified earlier)

route_lengths_regrouped = []

for i in route_lengths:
    if int(i) == 1 or int(i) == 2:
        route_lengths_regrouped.append("1-2")
    if int(i) >= 3 and int(i) <= 5:
        route_lengths_regrouped.append("3-5")
    if int(i) >= 6:
        route_lengths_regrouped.append("6+")
    if int(i) == 0:
        route_lengths_regrouped.append("Unsolved")

print(f"Regrouped route lengths:\n {Counter(route_lengths_regrouped).keys()} \n {Counter(route_lengths_regrouped).values()}")


# ECFP encoding
feature_list = []
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

for i in range(len(raw_smiles)):
    molecule_encoded = AllChem.MolFromSmiles(raw_smiles[i])
    feature_list.append(np.array(AllChem.GetMorganFingerprintAsBitVect(molecule_encoded, radius = R, nBits = L, useFeatures = use_features, useChirality = use_chirality)))

# print("Molecules encoded. Size of feature list: ", len(feature_list), len(feature_list[0]), feature_list[:5])
print(f"Molecules encoded. Feature size: {np.array(feature_list).shape}. Start fitting...")

# fit with XGBoost Classifier 
X, y = np.array(feature_list), np.array(route_lengths_regrouped)

# @TODO: This may need to be fixed: https://towardsdatascience.com/straightforward-stratification-bb0dcfcaf9ef 
# also: https://github.com/aiqc/aiqc 
#@TODO: Add saved train/test to xgboost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)

# model = XGBDistribution(
#     distribution="normal",
#     n_estimators=500,
#     early_stopping_rounds=10
# )

model = XGBClassifier()
# model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# LabelEncoder magic - otherwise it throws ValueError: Invalid classes inferred from unique values of `y`
le = LabelEncoder()
y_train = le.fit_transform(y_train)

model.fit(X_train, y_train, verbose=2)

print("Fitting finished. Calculating metrics in a different thread...")

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
make_confusion_matrix(confusion_matrix_test, categories=categories, figsize=(15, 15), filename=figures_dir + experiment_name + "_confusion_matrix_test.png")
# make_confusion_matrix(confusion_matrix_train, categories=categories, figsize=(15, 15), filename=figures_dir + experiment_name + "_confusion_matrix_train.png")
print("Plotting mosaic graph...")
nclass_classification_mosaic_plot(n_classes=len(confusion_matrix_test.tolist()), results=confusion_matrix_test.tolist(), filename=figures_dir + experiment_name + "_mosaic_test.png")
# nclass_classification_mosaic_plot(n_classes=len(confusion_matrix_train.tolist()), results=confusion_matrix_train.tolist(), filename=figures_dir + experiment_name + "_mosaic_train.png")

if calc_metric_thread.is_alive():
    print("Main thread execution finished but metric calculation is not finished yet. Waiting for it to finish...")
    calc_metric_thread.join()


# mean, std = preds.loc, preds.scale
# print("Fitting complete.\n mean: ", mean, " std: ", std, "\nPlotting graph...")
# plot_residuals(y_true=y_test, y_pred=preds.loc, y_err=preds.scale)

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


 