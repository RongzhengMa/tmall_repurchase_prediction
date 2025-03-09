import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

train_df = pd.read_csv("data/train_set.csv")
test_df = pd.read_csv("data/test_set.csv")

# 2. DATA
drop_cols = ["user_id", "merchant_id"]
X_train = train_df.drop(columns=drop_cols + ["label"])  
y_train = train_df["label"]  
X_test = test_df.drop(columns=drop_cols + ["label"])
y_test = test_df["label"]

# 3. Normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. training KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k, weights='distance')  # 使用加权距离
knn.fit(X_train_scaled, y_train)

# 5. prediction
y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 6.GridSearchCV
#param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
#grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1', cv=5)
#grid_search.fit(X_train_scaled, y_train)

#print("Best parameters:", grid_search.best_params_)
#print("Best F1 Score:", grid_search.best_score_)

#7.visualization


# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], labels=["Not Repeat", "Repeat"])
plt.yticks([0, 1], labels=["Not Repeat", "Repeat"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# 在方块中添加数值
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.show()

# 获取 KNN 预测的概率值（部分 KNN 版本不支持 predict_proba）
y_probs = knn.predict_proba(X_test_scaled)[:, 1]  # 只取正例 (label=1) 的概率

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # 参考线
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()



# 计算 Precision-Recall 曲线
precision, recall, _ = precision_recall_curve(y_test, y_probs)

# 绘制 Precision-Recall 曲线
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="red", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()


metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

# 绘制条形图
plt.figure(figsize=(6, 5))
plt.bar(metrics, values, color=["blue", "green", "orange", "red"], alpha=0.7)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)  # 0 到 1 之间
plt.title("Classification Metrics for KNN")
plt.show()
