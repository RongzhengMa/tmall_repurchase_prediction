import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


train_path = "data/train_set.csv"
test_path = "data/test_set.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

features = [
    "age_range", "gender", "total_logs", "unique_item_ids", "categories",
    "browse_days", "one_clicks", "shopping_carts", "purchase_times", "favourite_times"
]

X = train_df[features]
y = train_df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

logit_model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=1)
logit_model.fit(X_train_scaled, y_train)

y_val_pred = logit_model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy)

cv_scores = cross_val_score(logit_model, X_train_scaled, y_train, cv=3)
print("Cross-validation Accuracy:", cv_scores.mean())

X_train_full_scaled = scaler.fit_transform(X)
logit_model.fit(X_train_full_scaled, y)

print("Logit Coefficients:\n", logit_model.coef_)
print("Logit Intercept:\n", logit_model.intercept_)

X_test = test_df[features]
X_test_scaled = scaler.transform(X_test)

test_df["predicted_label"] = logit_model.predict(X_test_scaled)
test_df["predicted_proba"] = logit_model.predict_proba(X_test_scaled)[:, 1] 

test_df.to_csv("logit_test_set_with_predictions.csv", index=False)
print("Predictions saved to logit_test_set_with_predictions.csv")



y_test = test_df["label"]
y_pred = logit_model.predict(X_test_scaled) 
y_probs = logit_model.predict_proba(X_test_scaled)[:, 1]  

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Repeat", "Repeat"], yticklabels=["Not Repeat", "Repeat"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC Curve)")
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="red", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()



