import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score
import matplotlib.pyplot as plt

# load data
train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')

# train and validation
features = [col for col in train_df.columns if col not in ['user_id', 'merchant_id', 'label']]
X = train_df[features]
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


# 5. set XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

# param_distributions
param_distributions = {
    'n_estimators': [100],          
    'max_depth': [3],                 
    'learning_rate': [0.0422223],     
    'min_child_weight': [15],         
    'subsample': [0.75],           
    'colsample_bytree': [1.0]          
}

# RandomizedSearchCV 
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,           
    scoring='roc_auc',
    cv=3,                 
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best parameters found:", random_search.best_params_)

best_xgb = random_search.best_estimator_
y_val_pred = best_xgb.predict(X_val)
print("Validation evaluation:")
print(classification_report(y_val, y_val_pred))

val_pred_prob = best_xgb.predict_proba(X_val)[:, 1]
precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, val_pred_prob)
# cal F1-score
f1_scores = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
best_threshold = thresholds_val[np.argmax(f1_scores)]
print("Best threshold based on validation set:", best_threshold)

# 10. Reuse the whole train set
best_xgb.fit(X, y)

# 11. predict on test set
X_test = test_df[features]
y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_pred_prob > best_threshold).astype(int)

if 'label' in test_df.columns:
    print("Test evaluation (after threshold tuning):")
    print(classification_report(test_df['label'], y_pred_adjusted))
    print("Test Accuracy:", accuracy_score(test_df['label'], y_pred_adjusted))

# create CSV file
test_df['prediction'] = y_pred_adjusted
test_df['prediction_proba'] = y_pred_prob

output_csv = 'xgboost_predictions.csv'
test_df.to_csv(output_csv, index=False)
print("Predictions saved to:", output_csv)


if 'label' in test_df.columns:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(test_df['label'], y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Test Set')
    plt.legend(loc="lower right")
    plt.show()
