import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_score, recall_score, confusion_matrix,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

file1_path = "xgboost_predictions.csv"
file2_path = "h2o_automl_predictions.csv"
file3_path = "gradient_boosting_predictions.csv"
file4_path = "randomforest.csv"
file5_path = "logit_test_set_with_predictions.csv"
df_xgb = pd.read_csv(file1_path)
df_h2o = pd.read_csv(file2_path)
df_gdbt = pd.read_csv(file3_path)
df_rf = pd.read_csv(file4_path)
df_logit = pd.read_csv(file5_path)


merge_keys = ["user_id", "merchant_id"]

df_xgb = df_xgb.rename(columns={"prediction": "prediction_xgb", "prediction_proba": "proba_xgb"})
df_h2o = df_h2o.rename(columns={"predicted_label": "prediction_h2o", "p1": "proba_h2o"})
df_gdbt = df_gdbt.rename(columns={"prediction": "prediction_gdbt", "prediction_proba": "proba_gdbt"})
df_rf = df_rf.rename(columns={"predicted_label": "prediction_rf", "predicted_probability": "proba_rf"})
df_logit = df_logit.rename(columns={"predicted_label": "prediction_logit", "predicted_proba": "proba_logit"})

df_merged = df_xgb[merge_keys + ["label", "prediction_xgb", "proba_xgb"]]
df_merged = df_merged.merge(df_h2o[merge_keys + ["prediction_h2o", "proba_h2o"]], on=merge_keys, how="inner")
df_merged = df_merged.merge(df_gdbt[merge_keys + ["prediction_gdbt", "proba_gdbt"]], on=merge_keys, how="inner")
df_merged = df_merged.merge(df_rf[merge_keys + ["prediction_rf", "proba_rf"]], on=merge_keys, how="inner")
df_merged = df_merged.merge(df_logit[merge_keys + ["prediction_logit", "proba_logit"]], on=merge_keys, how="inner")

accuracy = {
    "XGBoost": accuracy_score(df_merged["label"], df_merged["prediction_xgb"]),
    "H2O AutoML": accuracy_score(df_merged["label"], df_merged["prediction_h2o"]),
    "Gradient Boosting": accuracy_score(df_merged["label"], df_merged["prediction_gdbt"]),
    "Random Forest": accuracy_score(df_merged["label"], df_merged["prediction_rf"]),
    "Logistic Regression": accuracy_score(df_merged["label"], df_merged["prediction_logit"])
}

aucpr = {
    "XGBoost": average_precision_score(df_merged["label"], df_merged["proba_xgb"]),
    "H2O AutoML": average_precision_score(df_merged["label"], df_merged["proba_h2o"]),
    "Gradient Boosting": average_precision_score(df_merged["label"], df_merged["proba_gdbt"]),
    "Random Forest": average_precision_score(df_merged["label"], df_merged["proba_rf"]),
    "Logistic Regression": average_precision_score(df_merged["label"], df_merged["proba_logit"]),
}

precision = {
    "XGBoost": precision_score(df_merged["label"], df_merged["prediction_xgb"]),
    "H2O AutoML": precision_score(df_merged["label"], df_merged["prediction_h2o"]),
    "Gradient Boosting": precision_score(df_merged["label"], df_merged["prediction_gdbt"]),
    "Random Forest": precision_score(df_merged["label"], df_merged["prediction_rf"]),
    "Logistic Regression": precision_score(df_merged["label"], df_merged["prediction_logit"])
}

recall = {
    "XGBoost": recall_score(df_merged["label"], df_merged["prediction_xgb"]),
    "H2O AutoML": recall_score(df_merged["label"], df_merged["prediction_h2o"]),
    "Gradient Boosting": recall_score(df_merged["label"], df_merged["prediction_gdbt"]),
    "Random Forest": recall_score(df_merged["label"], df_merged["prediction_rf"]),
    "Logistic Regression": recall_score(df_merged["label"], df_merged["prediction_logit"])
}

auc_scores = {
    "XGBoost": roc_auc_score(df_merged["label"], df_merged["proba_xgb"]),
    "H2O AutoML": roc_auc_score(df_merged["label"], df_merged["proba_h2o"]),
    "Gradient Boosting": roc_auc_score(df_merged["label"], df_merged["proba_gdbt"]),
    "Random Forest": roc_auc_score(df_merged["label"], df_merged["proba_rf"]),
    "Logistic Regression": roc_auc_score(df_merged["label"], df_merged["proba_logit"]),
}

for model, acc in accuracy.items():
    print(f"{model} Accuracy: {acc:.4f}")

for model, auc_value in aucpr.items():
    print(f"{model} AUCPR: {auc_value:.4f}")

for model, prec in precision.items():
    print(f"{model} Precision: {prec:.4f}")

for model, rec in recall.items():
    print(f"{model} Recall: {rec:.4f}")

for model, auc in auc_scores.items():
    print(f"{model} AUC: {auc:.4f}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    filename = f"figures/{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(model_name+" finished")

plot_confusion_matrix(df_merged["label"], df_merged["prediction_xgb"], "XGBoost")
plot_confusion_matrix(df_merged["label"], df_merged["prediction_h2o"], "H2O AutoML")
plot_confusion_matrix(df_merged["label"], df_merged["prediction_gdbt"], "Gradient Boosting")
plot_confusion_matrix(df_merged["label"], df_merged["prediction_rf"], "Random Forest")
plot_confusion_matrix(df_merged["label"], df_merged["prediction_logit"], "Logistic Regression")