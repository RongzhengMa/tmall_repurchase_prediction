from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
# Load data
train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')

x = train_df.drop(['user_id','merchant_id','label'],axis=1)
y = train_df['label']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state = 42)
params = {
    'n_estimators': 2000,  # 同 'n_estimators'
    'max_depth': 5,  # 同 'max_depth'
    'learning_rate': 0.01,  # 同 'learning_rate'
    'min_samples_leaf': 4,  # 同 'min_data_in_leaf'
    'subsample': 0.8,  # 同 "colsample_bytree"，但这是整个数据集的比例，而不是特征的比例
    'max_features': 0.8,  # 同 "colsample_bytree"，但这是特征的比例，GradientBoostingClassifier中没有直接对应的参数，这里假设是特征的比例
    'random_state': 42,  # 同 'seed'
}

# GradientBoostingClassifier
gbm = GradientBoostingClassifier(**params)

# train
gbm.fit(x_train, y_train)

#prediction
gbm_pred = gbm.predict(x_val)
gbm_proba = gbm.predict_proba(x_val)

# 
print('模型的评估报告：\n', classification_report(y_val, gbm_pred))
# ROC curve
fpr, tpr, thresholds = roc_curve(y_val, gbm_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# param_distributio
param_distributions = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

# 创建RandomizedSearchCV实例
random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=param_distributions,
                                   n_iter=20, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1, random_state=42)

# 执行随机搜索
random_search.fit(x_train, y_train)

# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)

# 使用最佳参数创建模型
best_gbm = random_search.best_estimator_

# 使用最佳模型进行预测
best_gbm_pred = best_gbm.predict(x_val)
best_gbm_proba = best_gbm.predict_proba(x_val)

# 打印分类报告
print('Best model evaluation report:\n', classification_report(y_val, best_gbm_pred))
# 绘制ROC曲线
best_fpr, best_tpr, best_thresholds = roc_curve(y_val, best_gbm_proba[:, 1])
best_roc_auc = auc(best_fpr, best_tpr)
plt.figure()
plt.plot(best_fpr, best_tpr, label='ROC curve (area = %0.2f)' % best_roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Best Model')
plt.legend(loc="lower right")
plt.show()

#csv
x_test_final = test_df.drop(['user_id', 'merchant_id', 'label'], axis=1)

test_pred = best_gbm.predict(x_test_final)
test_pred_proba = best_gbm.predict_proba(x_test_final)[:, 1]

test_df['prediction'] = test_pred
test_df['prediction_proba'] = test_pred_proba

output_csv = 'gradient_boosting_predictions.csv'
test_df.to_csv(output_csv, index=False)
print("Predictions saved to:", output_csv)