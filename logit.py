import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
