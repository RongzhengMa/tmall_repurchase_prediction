import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1', cv=5)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)