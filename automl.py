import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import json

h2o.init(max_mem_size='16G')

train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')

features = [col for col in train_df.columns if col not in ['user_id', 'merchant_id', 'label']]
target = 'label'

train_hf = h2o.H2OFrame(train_df)
test_hf = h2o.H2OFrame(test_df)


train_hf[target] = train_hf[target].asfactor()


train_hf, val_hf = train_hf.split_frame(ratios=[0.8], seed=42)

num_zeros = val_hf[val_hf[target] == "0", :].nrow
num_ones = val_hf[val_hf[target] == "1", :].nrow
scale_pos_weight = num_zeros / num_ones

aml = H2OAutoML(
    max_runtime_secs=3600,         
    exclude_algos=["DeepLearning"], 
    balance_classes=True,         
    class_sampling_factors=[1, scale_pos_weight],
    stopping_metric="AUCPR",   
    seed=42,
    nfolds=3,          
    sort_metric="aucpr",
)

aml.train(x=features, y=target, training_frame=train_hf, validation_frame=val_hf)


lb = aml.leaderboard


best_model = aml.leader
print("Best Model:", best_model) 

if val_hf:
    valid_perf = best_model.model_performance(val_hf)
    print(f"Validation AUC-PR: {valid_perf.aucpr():.4f}")


test_hf = h2o.H2OFrame(test_df)
test_perf = best_model.model_performance(test_hf)
print("\nTest Set Performance:")
print(f"AUC: {test_perf.auc():.4f}")
print(f"AUC-PR: {test_perf.aucpr():.4f}")

valid_preds = best_model.predict(val_hf)
val_probs = valid_preds['p1'].as_data_frame().values.flatten()
val_target = val_hf[target].as_data_frame().values.flatten()

precision, recall, thresholds = precision_recall_curve(val_target, val_probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nUsing classification threshold: {best_threshold}")

preds = best_model.predict(test_hf)
test_hf = test_hf.cbind(preds)
test_hf['predicted_label'] = (test_hf['p1'] > best_threshold).ifelse(1, 0)

output_df = test_hf.as_data_frame()
output_csv = 'h2o_automl_predictions.csv'
output_df.to_csv(output_csv, index=False)
print(f"\nPredictions saved to: {output_csv}")

fpr, tpr, _ = roc_curve(output_df['label'], output_df['p1'])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("figures/Figure_13.png", dpi=300, bbox_inches="tight")

h2o.cluster().shutdown(prompt=False)