from plotter import make_confusion_matrix, nclass_classification_mosaic_plot
import pandas as pd
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# experiment_name = "50ktest2"
experiment_name = "1k"
experiment_mode = 'multiclass' # 'regression' or 'multiclass' 
results_dir = "results/" + experiment_name + "/"
figures_dir = "figures/" + experiment_name + "/"
chemprop_train_csv = "gnn/data/" + experiment_name + "_train.csv"
chemprop_test_csv = "gnn/data/" + experiment_name + "_test.csv"
chemprop_prediction = "gnn/data/" + experiment_name + "_" + experiment_mode + "_prediction"
chemprop_model_dir = "gnn/model/" + experiment_name + "_" + experiment_mode 
file_name = "gnn/data/" + experiment_name + "_" + experiment_mode + "_prediction"
test_file = "gnn/data/" + experiment_name + "_test.csv"


# Read the CSV file and import data into a dataframe
df = pd.read_csv(file_name)

# regrouped route lengths into 4 classes -- for multiclass tasks
# 0. unsolved (length == 0 as specified earlier)
# 1. 1-2 steps
# 2. 3-5 steps
# 3. 6+ steps

if experiment_mode == 'multiclass':
    # 1. Create a new column "max_probability" with the maximum values from route_length_class_0 to route_length_class_3
    df['max_probability'] = df[['route_length_class_0', 'route_length_class_1', 'route_length_class_2', 'route_length_class_3']].max(axis=1)

    # 2. Create a new column "max_predicted_class" with the class that gives the maximum possibility
    df['max_predicted_class'] = df[['route_length_class_0', 'route_length_class_1', 'route_length_class_2', 'route_length_class_3']].idxmax(axis=1).str[-1:].astype(int)

    # 3. Create a new column "correct_prediction" with True/False based on the comparison of "route_length" and "max_predicted_class"
    df['correct_prediction'] = df['route_length'] == df['max_predicted_class']

    # 4. Print the specified columns for the first 50 entries
    print(f"Top 50 lines of result from {experiment_name}_{experiment_mode}\n", df[['smile', 'route_length', 'max_probability', 'max_predicted_class', 'correct_prediction']].head(50))
    confusion_matrix_test = metrics.confusion_matrix(df['route_length'], df['max_predicted_class'])
    categories = ['unsolved', '1-2', '3-5', '6+']
    matrix_plot_path = figures_dir + experiment_name + "_gnn_" + experiment_mode + "_confusion_matrix.png"
    matrix_plot_title = experiment_name + "_gnn_" + experiment_mode + "_confusion_matrix"
    roc_graph_path = figures_dir + experiment_name + "_gnn_" + experiment_mode + "_ROC_graph.png"
    roc_graph_title = experiment_name + "_gnn_" + experiment_mode + "_ROC_graph"

    print(f"\nConfusion matrix for {experiment_name}_{experiment_mode}: \n", confusion_matrix_test)
    print(f"plotting {matrix_plot_title}...")
    make_confusion_matrix(confusion_matrix_test, categories=categories, figsize=(15, 15), title=matrix_plot_title, filename=matrix_plot_path)

    
    def calculate_confusion_matrix_values(y_true, y_pred):
        confusion_matrix_test = confusion_matrix(y_true, y_pred)
        tn = confusion_matrix_test[0, 0]
        fp = confusion_matrix_test[0, 1]
        fn = confusion_matrix_test[1, 0]
        tp = confusion_matrix_test[1, 1]
        return tn, fp, fn, tp

    recall = metrics.recall_score(df['route_length'], df['max_predicted_class'], average='weighted')
    f1_score = metrics.f1_score(df['route_length'], df['max_predicted_class'], average='weighted')
    tn, fp, fn, tp = calculate_confusion_matrix_values(df['route_length'], df['max_predicted_class'])

    false_positive_rate = fp / (fp + tn)


    y_true_binary = label_binarize(df['route_length'], classes=[0, 1, 2, 3])
    y_pred_binary = label_binarize(df['max_predicted_class'], classes=[0, 1, 2, 3])

    average_precision = metrics.average_precision_score(y_true_binary, y_pred_binary, average='weighted')

    fpr, tpr, thresholds = metrics.roc_curve(y_true_binary.ravel(), y_pred_binary.ravel())
    auc_roc = metrics.auc(fpr, tpr)

    print(f"plotting {roc_graph_title}...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='AUC ROC = {:.2f}'.format(auc_roc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    metrics_text = f"Recall: {recall:.2f}, F1 Score: {f1_score:.2f}, AUC ROC: {auc_roc:.2f}, Average Precision: {average_precision:.2f}"
    plt.title(roc_graph_title+'\n' + metrics_text)
    plt.legend(loc='lower right')
    plt.savefig(roc_graph_path)

if experiment_mode == 'regression':
    df2 = pd.read_csv(test_file)
    df['route_length_predicted'] = df['route_length']
    df['route_length_truth'] = df2['route_length']
    print(f"Top 50 lines of result from {experiment_name}_{experiment_mode}\n", df[['smile', 'route_length_predicted', 'route_length_truth']].head(50))


    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(df['route_length_truth'], df['route_length_predicted']))

    # Print the RMSE value
    print("RMSE:", rmse)

    # Calculate the Mean Absolute Error
    mae = mean_absolute_error(df['route_length_truth'], df['route_length_predicted'])

    # Print the Mean Absolute Error value
    print("Mean Absolute Error:", mae)

    scatter_plot_path = figures_dir + experiment_name + "_gnn_" + experiment_mode + "_scatter_plot.png"
    scatter_plot_title = experiment_name + "_gnn_" + experiment_mode + "_scatter_plot"

    print(f"plotting {scatter_plot_title}...")

    plt.figure(figsize=(8, 6))
    plt.scatter(df['route_length_truth'], df['route_length_predicted'], color='blue', label='Data Points')
    plt.plot(df['route_length_truth'], df['route_length_truth'], color='red', label='Fitted Line')
    plt.xlabel('True Route Length')
    plt.ylabel('Predicted Route Length')
    plt.title(scatter_plot_title + f"\nRMSE: {rmse:.2f}, MAE: {mae:.2f}")
    plt.legend()
    plt.savefig(scatter_plot_path)

print("Analysis done!")
