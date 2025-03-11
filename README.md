# tmall_repurchase_prediction

# Logistic Regression for Repurchase Prediction #

**Processing Logic:**
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
Cross-validation Accuracy: 0.938<br>

Confusion Matrix
![image](https://github.com/RongzhengMa/tmall_repurchase_prediction/blob/main/figures/Figure_10.png) <br>

•	High accuracy but imbalanced predictions.
•	Many false negatives (Repeat customers predicted as Not Repeat).

ROC Curve
![image](https://github.com/RongzhengMa/tmall_repurchase_prediction/blob/main/figures/Figure_11.png) <br>
•	AUC = 0.62, indicating moderate predictive ability.

**Limitations:**
1.	Class Imbalance: Most customers do not repurchase, leading to poor recall. <br>

2.	Model Selection: Logistic regression is linear. <br>


