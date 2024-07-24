import os
from confusion_matrix_plotter import make_confusion_matrix
import pandas as pd
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_curve, auc, mean_absolute_error
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from experiment_config import Experiment

experiment_name = ""
experiment_mode = "" 
results_dir = ""
figures_dir = ""
file_name = ""
test_file = ""
xgboost_prediction = ""
test_set = ""
inference_option = ""
inference_name = ""
df = None

def calculate_regression_metrics(df):
    rmse = np.sqrt(mean_squared_error(y_true=df['route_length_truth'], y_pred=df['route_length_predicted']))
    print("RMSE:", rmse)

    mae = mean_absolute_error(y_true=df['route_length_truth'], y_pred=df['route_length_predicted'])
    print("Mean Absolute Error:", mae)

    r_squared = metrics.r2_score(y_true=df['route_length_truth'], y_pred=df['route_length_predicted'])
    print("R-squared:", r_squared)

    return [rmse, mae, r_squared]


def make_box_plot(df, metrics: list):
    rmse, mae, r_squared = metrics[0:3]
    box_plot_path = figures_dir + experiment_name + "_xg_boost_" + experiment_mode + "_box_plot.png"
    box_plot_title = experiment_name + "_xg_boost_" + experiment_mode + "_box_plot"
        
    print(f"plotting {box_plot_title}...")
        
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['route_length_truth'], y=df['route_length_predicted'], showfliers=False)
    plt.xlabel('True Route Length')
    plt.ylabel('Predicted Route Length')
    plt.title(box_plot_title + f"\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.2f}")
    plt.savefig(box_plot_path)


def make_reg_plot(df, metrics: list):
    rmse, mae, r_squared = metrics[0:3]
    table_path = figures_dir + experiment_name + "_xg_boost_" + experiment_mode + "_regression_plot.png"
    table_title = experiment_name + "_xg_boost_" + experiment_mode + "_regression_plot"
        
    print(f"plotting {table_title}...")
        
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df['route_length_truth'], y=df['route_length_predicted'])
    plt.xlabel('True Route Length')
    plt.ylabel('Predicted Route Length')
    plt.title(table_title + f"\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r_squared:.2f}")
    if not os.path.exists(os.path.dirname(table_path)):
        os.makedirs(os.path.dirname(table_path))
    plt.savefig(table_path)


def calculate_confusion_matrix_values(y_true, y_pred):
    confusion_matrix_test = confusion_matrix(y_true, y_pred)
    tn = confusion_matrix_test[0, 0]
    fp = confusion_matrix_test[0, 1]
    fn = confusion_matrix_test[1, 0]
    tp = confusion_matrix_test[1, 1]
    return tn, fp, fn, tp


def set_parameters(experiment: Experiment):
    global experiment_name, experiment_mode, results_dir, figures_dir, file_name, test_file, xgboost_prediction, test_set, df, inference_option, inference_name

    experiment_name = experiment.experiment_name
    experiment_mode = experiment.experiment_mode
    results_dir = experiment.results_dir
    figures_dir = experiment.figures_dir
    file_name = experiment.chemprop_prediction
    test_file = experiment.test_set
    xgboost_prediction = experiment.xgboost_prediction
    test_set = experiment.test_set
    inference_option = experiment.inference_option
    inference_name = experiment.inference_name
    df = pd.read_csv(xgboost_prediction)

    if inference_option:
        experiment_name += "_inference_" + inference_name


def plot_residuals(y_true, y_pred, y_err, filename=None, title=None):
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
    ax.set_title(title)
    plt.savefig(filename)


# regrouped route lengths into 4 classes -- for multiclass tasks
# 0. unsolved (length == 0 as specified earlier)
# 1. 1-2 steps
# 2. 3-5 steps
# 3. 6+ steps


def run_xg_boost_result_analysis():
    df2 = pd.read_csv(test_set)
    df['route_length_predicted'] = df['route_length']
    df['route_length_truth'] = df2['route_length']
    
    print(f"================================================== Starting XGBoost Result analysis for {experiment_name} {experiment_mode}... ==================================================")
    if experiment_mode == 'multiclass':
        print(f"Top 50 lines of result from {experiment_name}_{experiment_mode}\n", df[['smiles', 'route_length_predicted', 'route_length_truth']].head(50))
        confusion_matrix_test = metrics.confusion_matrix(y_pred=df['route_length'], y_true=df['route_length_truth'])
        categories = ['unsolved', '1-2', '3-5', '6+']
        matrix_plot_path = figures_dir + experiment_name + "_xg_boost_" + experiment_mode + "_confusion_matrix.png"
        matrix_plot_title = experiment_name + "_xg_boost_" + experiment_mode + "_confusion_matrix"
        roc_graph_path = figures_dir + experiment_name + "_xg_boost_" + experiment_mode + "_ROC_graph.png"
        roc_graph_title = experiment_name + "_xg_boost_" + experiment_mode + "_ROC_graph"

        print(f"\nConfusion matrix for {experiment_name}_{experiment_mode}: \n", confusion_matrix_test)
        print(f"plotting {matrix_plot_title}...")
        make_confusion_matrix(confusion_matrix_test, categories=categories, figsize=(15, 15), title=matrix_plot_title, filename=matrix_plot_path)

        recall = metrics.recall_score(y_pred=df['route_length'], y_true=df['route_length_truth'], average='weighted')
        f1_score = metrics.f1_score(y_pred=df['route_length'], y_true=df['route_length_truth'], average='weighted')
        mcc_score = metrics.matthews_corrcoef(y_pred=df['route_length'], y_true=df['route_length_truth'])
        accuracy = metrics.accuracy_score(y_pred=df['route_length'], y_true=df['route_length_truth'])
        tn, fp, fn, tp = calculate_confusion_matrix_values(y_pred=df['route_length'], y_true=df['route_length_truth'])

        false_positive_rate = fp / (fp + tn)

        y_true_binary = label_binarize(df['route_length'], classes=[0, 1, 2, 3])
        y_pred_binary = label_binarize(df['route_length_truth'], classes=[0, 1, 2, 3])

        average_precision = metrics.average_precision_score(y_true=y_true_binary, y_score=y_pred_binary, average='weighted')

        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true_binary.ravel(), y_score=y_pred_binary.ravel())
        auc_roc = metrics.auc(fpr, tpr)

        print(f"plotting {roc_graph_title}...")
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='AUC ROC = {:.2f}'.format(auc_roc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        metrics_text = f"Recall: {recall:.2f}, F1 Score: {f1_score:.2f}, AUC ROC: {auc_roc:.2f}, Average Precision: {average_precision:.2f}, MCC: {mcc_score:.2f}, Accuracy: {accuracy:.2f}"
        plt.title(roc_graph_title+'\n' + metrics_text)
        plt.legend(loc='lower right')
        plt.savefig(roc_graph_path)

        # Create a table with the metrics
        table_path = figures_dir + experiment_name + "_xgboost_" + experiment_mode + "_classification_table.png"
        table_title = experiment_name + "_xgboost_" + experiment_mode + "_classification_table"
        print(f"plotting {table_title}...")
        classification_table = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'MCC', 'Precision', 'Recall'],
            'Value': [round(accuracy, 3), round(f1_score, 3), round(mcc_score, 3), round(average_precision, 3), round(recall, 3)]
        })
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        table = plt.table(cellText=classification_table.values, colLabels=classification_table.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        for i in range(len(classification_table.columns)):
            table[0, i].set_facecolor('grey')
        for i in range(len(classification_table)):
            table[i + 1, 0].set_facecolor('lightgrey')
        plt.title(table_title, y=0.8)  # Adjust the value of 'y' to reduce the distance between the title and the table
        plt.savefig(table_path)

    if experiment_mode == 'regression':
        print(f"Top 50 lines of result from {experiment_name}_{experiment_mode}\n", df[['smiles', 'route_length_predicted', 'route_length_truth']].head(50))

        regression_metrics = calculate_regression_metrics(df)
        make_box_plot(df, regression_metrics)
        make_reg_plot(df, regression_metrics)

    print(f"================================================== Finished XGBoost Result Analysis for {experiment_name} {experiment_mode}... ==================================================")


if __name__ == "__main__":
    run_xg_boost_result_analysis()