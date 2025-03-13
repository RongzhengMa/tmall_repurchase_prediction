import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score

# Load data
train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')
x = train_df.drop(['user_id', 'merchant_id', 'label'], axis=1)
y = train_df['label']

# train and test
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# param_distributio
param_distributions = {
    'n_estimators': [1200],            
    'max_depth': [4],                       
    'learning_rate':[0.011421052631578946],          
    'min_samples_leaf': [18],               
    'subsample': [0.7],                  
    'max_features': [0.9]                
}

# RandomizedSearchCV 
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,          
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(x_train, y_train)

# best ROC AUC
print("Best parameters found: ", random_search.best_params_)
print("Best ROC AUC on validation (CV):", random_search.best_score_)

# evaluation
best_gbm = random_search.best_estimator_
y_val_pred = best_gbm.predict(x_val)
y_val_proba = best_gbm.predict_proba(x_val)[:, 1]

print("Validation classification report:\n", classification_report(y_val, y_val_pred))
print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
roc_auc_value = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_value)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best Model (Validation)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# CSV file
x_test_final = test_df.drop(['user_id', 'merchant_id', 'label'], axis=1)
test_pred = best_gbm.predict(x_test_final)
test_pred_proba = best_gbm.predict_proba(x_test_final)[:, 1]

test_df['prediction'] = test_pred
test_df['prediction_proba'] = test_pred_proba

output_csv = 'gradient_boosting_predictions.csv'
test_df.to_csv(output_csv, index=False)
print("Predictions saved to:", output_csv)