# tmall_repurchase_prediction

## Logistic Regression for Repurchase Prediction

**Processing Logic:**<br>
The goal of this project is to use a logistic regression model to predict whether a customer will repurchase an item based on various features. The process involves:<br>
1.	Data Preparation: Loading and preprocessing data from train_set.csv and test_set.csv.<br>
2.	Feature Selection: Using customer behavior metrics as predictive features.<br>
3.	Model Training: Splitting data into training and validation sets, scaling, and training a logistic regression model.<br>
4.	Evaluation: Assessing model performance using validation accuracy, confusion matrix, ROC curve, and Precision-Recall curve.<br>
5.	Prediction on Test Data: Making final predictions and saving the results.<br>

**Key Functions:** <br>
`strain_test_split(X, y, test_size=0.2, random_state=42, stratify=y): `: Splits the dataset into training and validation sets. test_size=0.2: Allocates 20% of the data for validation<br>
`scaler = StandardScaler()`: Standardizes numerical features by removing the mean and scaling to unit variance. <br>
`logit_model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=1)`: Uses logistic regression to predict binary outcomes (0 or 1). max_iter=1000: Increases the number of iterations to ensure convergence.<br>
`test_df["predicted_label"] = logit_model.predict(X_test_scaled)`
`test_df["predicted_proba"] = logit_model.predict_proba(X_test_scaled)[:, 1]`: Predicts purchase behavior on new customers. <br>
`sns.heatmap`:Displays a heatmap representation of the confusion matrix. <br>
`fpr, tpr, _ = roc_curve(y_val, y_probs)`
`roc_auc = auc(fpr, tpr)`:Calculates AUC (Area Under Curve)<br>


**Results & Figures:** <br>
Validation Accuracy: 0.938<br>
Cross-validation Accuracy: 0.939<br>

Confusion Matrix
![image](https://github.com/RongzhengMa/tmall_repurchase_prediction/blob/main/figures/Figure_10.png) <br>

•	High accuracy but imbalanced predictions.
•	Many false negatives (Repeat customers predicted as Not Repeat).

ROC Curve
![image](https://github.com/RongzhengMa/tmall_repurchase_prediction/blob/34-logit_figure/figures/Figure11.png)) <br>
•	AUC = 0.62, indicating moderate predictive ability.

**Limitations:**
1.	Class Imbalance: Most customers do not repurchase, leading to poor recall. <br>

2.	Model Selection: Logistic regression is linear. <br>


## GBDT

### Overview
Gradient Boosting Decision Trees (GBDT) is an ensemble learning technique that builds multiple weak decision trees sequentially, with each new tree correcting the errors of the previous ones. It is particularly effective for structured data and can capture complex patterns in customer behavior.

### Data Processing & Modeling Steps

- **Feature Engineering:**
  - Removed irrelevant identifiers such as `user_id`, `merchant_id`, and the target label.
  - Incorporated behavioral and transactional features extracted from user logs.

- **Data Splitting:**
  - Training Set: 80%
  - Validation Set: 20%

- **Hyperparameter Optimization:**
  - Employed **RandomizedSearchCV** with 3-fold cross-validation to fine-tune model parameters.
  - Optimal parameters identified:
  ```
  n_estimators = 1200
  learning_rate = 0.0114
  max_depth = 4
  min_samples_leaf = 18
  subsample = 0.7
  max_features = 0.9
  ```

### Model Performance

- **Cross-validation ROC AUC:** `0.6227`
- **Validation ROC AUC:** `0.6105`

#### Classification Report (Validation Set):

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0 (Not Returning) | 0.94 | 1.00 | 0.97 |
| 1 (Returning) | 0.15 | 0.00 | 0.00 |

- The model effectively identifies non-returning customers but struggles with classifying returning customers due to the **severe class imbalance**.

### Limitations and Future Enhancements

- **Class Imbalance:**
  - The dataset is highly imbalanced, significantly affecting recall for returning customers.
  - Potential solutions include **SMOTE (Synthetic Minority Over-sampling Technique)** or **cost-sensitive learning**.

- **Feature Engineering:**
  - Incorporating additional behavioral, demographic, and historical purchase data could improve predictive power.

- **Alternative Models:**
  - Exploring **XGBoost** or **LightGBM** might yield better performance.

- **Ensemble Methods:**
  - Combining multiple models could enhance accuracy and robustness.

### Visualization

The ROC curve (`gbdt-roc.png`) illustrates the model’s performance. A **ROC AUC of 0.61** suggests limited predictive capability, indicating the need for further refinement and feature optimization.

## XGBoost

### Overview
Extreme Gradient Boosting (XGBoost) is a powerful and efficient implementation of gradient boosting, designed for high performance and scalability. It is particularly effective for structured data and is widely used for classification and regression tasks. XGBoost employs a combination of decision trees and boosting techniques to minimize error iteratively, making it an ideal choice for customer behavior prediction.

### Data Processing & Modeling Steps

- **Feature Engineering:**
  - Removed irrelevant identifiers such as `user_id`, `merchant_id`, and the target label.
  - Incorporated behavioral and transactional features extracted from user logs.

- **Data Splitting:**
  - Training Set: 80%
  - Validation Set: 20%

- **Hyperparameter Optimization:**
  - Employed **RandomizedSearchCV** with 3-fold cross-validation to fine-tune model parameters.
  - Optimal parameters identified:
  ```
  n_estimators = 100
  learning_rate = 0.0422
  max_depth = 3
  min_child_weight = 15
  subsample = 0.75
  colsample_bytree = 1.0
  ```

### Model Performance

- **Validation Performance:**
  - **Validation ROC AUC:** `0.63`
  - **Best threshold (Validation Set):** `0.5879`

#### Classification Report (Validation Set):

| Class             | Precision | Recall | F1-score |
| ----------------- | --------- | ------ | -------- |
| 0 (Not Returning) | 0.95      | 0.69   | 0.80     |
| 1 (Returning)     | 0.09      | 0.49   | 0.16     |

- The model demonstrates improved recall for returning customers compared to GBDT but still exhibits significant class imbalance issues.

#### Test Set Evaluation (After Threshold Tuning):

| Class             | Precision | Recall | F1-score |
| ----------------- | --------- | ------ | -------- |
| 0 (Not Returning) | 0.95      | 0.87   | 0.91     |
| 1 (Returning)     | 0.12      | 0.28   | 0.17     |

- **Test Accuracy:** `0.8336`
- The threshold tuning helped improve recall for returning customers but at the cost of reduced precision.

### Limitations and Future Enhancements

- **Class Imbalance:**
  - The dataset is highly imbalanced, affecting the recall of returning customers.
  - Techniques like **SMOTE**, **focal loss**, or adjusting `scale_pos_weight` could improve performance.

- **Feature Engineering:**
  - Additional features, such as customer demographics, browsing patterns, and purchase frequency, could enhance predictive power.

- **Alternative Models:**
  - Comparing with **GBDT**, **LightGBM**, or ensemble methods may lead to better performance.

### Visualization

The ROC curve (`xgboost_roc.png`) illustrates the model’s performance on the test set. A **ROC AUC of 0.63** indicates that while the model has improved over GBDT, further optimizations are needed for better prediction accuracy.



