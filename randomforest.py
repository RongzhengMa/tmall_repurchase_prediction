# -*- coding: utf-8 -*-
"""RandomForest.ipynb

This file is originally exexuted in Google Colab notebook and then exported to .py file. Some cell employed GPU capacity for huge grid search and cross validation workload.
If you cannot rerun the code in your local machine, please refer to the outputs in RandomForest.ipynb

## Data exploration

Explore the training data, focusing on the distribution of the 'label' column and determine to use AUC-PR,Precision,Recall,F1 Score as metrics
"""

## Data exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix,precision_recall_curve,auc
from sklearn.preprocessing import StandardScaler
import warnings
import tensorflow as tf
import torch
import os
import locale
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Load data
train_data = pd.read_csv('train_set.csv')
test_data = pd.read_csv('test_set.csv')

# Display basic info
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
train_data.head()

"""## Split Sets and Feature Scaling"""

# Split data into features and target
X = train_data.drop('label', axis=1)
y = train_data['label']

# Check if test data contains 'label' column and remove it if it does
if 'label' in test_data.columns:
    X_test = test_data.drop('label', axis=1)
else:
    X_test = test_data

# Now split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Explore the label distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=train_data)
plt.title('Distribution of Labels')
plt.show()

print(f"Positive examples: {sum(train_data['label'] == 1)}")
print(f"Negative examples: {sum(train_data['label'] == 0)}")
print(f"Positive ratio: {sum(train_data['label'] == 1) / len(train_data):.2%}")

# --- Feature Importance Visualization ---
# Fit a RandomForestClassifier to get feature importances
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)  # You can adjust parameters
rf_model.fit(X_train_scaled, y_train)  # Using scaled training data

# Get feature importances and sort them
importances = rf_model.feature_importances_
feature_names = X_train.columns  # Get feature names from original data
indices = np.argsort(importances)[::-1]  # Sort indices in descending order

# Create a bar plot of feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)  # Rotate x-axis labels
plt.xlim([-1, X_train.shape[1]])  # Adjust x-axis limits for better visibility
plt.tight_layout()
plt.show()

# Explain chosen metrics
print("\nEvaluation metrics to be used:")
print("1. AUC-PR: More sensitive to unbalanced small portion of positive cases")
print("2. Accuracy: Proportion of correct predictions")
print("3. Precision: Proportion of positive identifications that were actually correct")
print("4. Recall: Proportion of actual positives that were identified correctly")
print("5. F1-score: Harmonic mean of precision and recall")

"""## Performance Calculation Function"""

# Helper function to calculate AUC-PR
def calculate_auc_pr(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

"""## Setup GPU For Grid Search"""

# Check if GPU is available and identify GPU type
import tensorflow as tf
import torch
import os
import locale

# Check if GPU is enabled with TensorFlow
print("TensorFlow version:", tf.__version__)
print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))

# Check if GPU is enabled with PyTorch
print("\nPyTorch version:", torch.__version__)
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch CUDA device:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    print("CUDA capability:", torch.cuda.get_device_capability(0))
    print("GPU memory allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
    print("GPU memory reserved:", round(torch.cuda.memory_reserved(0)/1024**3, 1), "GB")

"""## Baseline Model with Default"""

# Establish baseline model
base_params = {
    'n_estimators': 30,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
baseline_model = RandomForestClassifier(**base_params)
baseline_model.fit(X_train_scaled, y_train)

# Evaluate on training set
y_train_pred_baseline = baseline_model.predict(X_train_scaled)
y_train_prob_baseline = baseline_model.predict_proba(X_train_scaled)[:, 1]

# Calculate training metrics
train_auc_pr = calculate_auc_pr(y_train, y_train_prob_baseline)
train_accuracy = balanced_accuracy_score(y_train, y_train_pred_baseline)
train_precision = precision_score(y_train, y_train_pred_baseline,average='weighted')
train_recall = recall_score(y_train, y_train_pred_baseline,average='weighted')
train_f1 = f1_score(y_train, y_train_pred_baseline,average='weighted')

# Plot confusion matrix for training set
cm_train = confusion_matrix(y_train, y_train_pred_baseline)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Baseline Model (Training Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Evaluate on validation set
y_val_pred_baseline = baseline_model.predict(X_val_scaled)
y_val_prob_baseline = baseline_model.predict_proba(X_val_scaled)[:, 1]

# Calculate validation metrics
val_auc_pr = calculate_auc_pr(y_val, y_val_prob_baseline)
val_accuracy = balanced_accuracy_score(y_val, y_val_pred_baseline)
val_precision = precision_score(y_val, y_val_pred_baseline,average='weighted')
val_recall = recall_score(y_val, y_val_pred_baseline,average='weighted')
val_f1 = f1_score(y_val, y_val_pred_baseline,average='weighted')

# Plot confusion matrix for validation set
cm_val = confusion_matrix(y_val, y_val_pred_baseline)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Baseline Model (Validation Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Compare metrics in a table
metrics_comparison = pd.DataFrame({
    'Metric': ['AUC-PR', 'Weighted Precision', 'Recall', 'F1-Score','Balanced Accuracy'],
    'Training': [train_auc_pr, train_precision, train_recall, train_f1,train_accuracy],
    'Validation': [val_auc_pr, val_precision, val_recall, val_f1,val_accuracy]
})

print("\nBaseline Model - Training vs. Validation Metrics:")
print(metrics_comparison)

"""## First Stage Grid Search for Parameter Selection

"""

# Parameter Selection Analysis for Imbalanced Data
print("Starting parameter analysis...")

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

# Initialize lists to store results
param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf','max_features']
param_values = {
    'n_estimators': [1, 10, 20, 30],
    'max_depth': [2, 5, 8, 15, None],
    'min_samples_split': [2, 10, 20, 35, 50],
    'min_samples_leaf': [1, 5, 10, 15, 30, 50],
    'max_features': ['sqrt', 'log2', None]
}

# Dictionary to store results - using global variables
global train_sensitivity_results, val_sensitivity_results
train_sensitivity_results = {}
val_sensitivity_results = {}

# Base model with default parameters
base_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'random_state': 42
}

# Define stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Helper function to calculate AUC-PR
def calculate_auc_pr(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

# For each parameter, vary its value while keeping others constant
print("Testing parameter sensitivities with cross-validation...")


for param_name in param_names:
    print(f"Analyzing parameter: {param_name}")
    train_param_scores = []
    val_param_scores = []

    for value in param_values[param_name]:
        # Create a copy of base parameters and update the current parameter
        current_params = base_params.copy()
        current_params[param_name] = value

        # Track scores across folds
        fold_train_scores = []
        fold_val_scores = []

        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train model with current parameters
            model = RandomForestClassifier(**current_params)
            model.fit(X_fold_train, y_fold_train)

            # Evaluate on training set
            y_fold_train_prob = model.predict_proba(X_fold_train)[:, 1]
            train_auc_pr = calculate_auc_pr(y_fold_train, y_fold_train_prob)
            fold_train_scores.append(train_auc_pr)

            # Evaluate on validation set
            y_fold_val_prob = model.predict_proba(X_fold_val)[:, 1]
            val_auc_pr = calculate_auc_pr(y_fold_val, y_fold_val_prob)
            fold_val_scores.append(val_auc_pr)

        # Calculate mean scores across folds
        mean_train_score = np.mean(fold_train_scores)
        mean_val_score = np.mean(fold_val_scores)

        # Store results
        train_param_scores.append((value, mean_train_score))
        val_param_scores.append((value, mean_val_score))

        print(f"  {param_name}={value}: Train={mean_train_score:.4f}, Val={mean_val_score:.4f}")

    # Store results for this parameter
    train_sensitivity_results[param_name] = train_param_scores
    val_sensitivity_results[param_name] = val_param_scores

# Verify that results were stored
print("\nResults stored successfully:")
print(f"Number of parameters analyzed: {len(train_sensitivity_results)}")
for param in train_sensitivity_results:
    print(f"  - {param}: {len(train_sensitivity_results[param])} values")

# To ensure the results persist, create a small verification dataframe
sensitivity_summary = pd.DataFrame({
    'Parameter': list(train_sensitivity_results.keys()),
    'Values_Tested': [len(train_sensitivity_results[p]) for p in train_sensitivity_results],
})
print("\nSensitivity Analysis Summary:")
print(sensitivity_summary)

print("Parameter sensitivity analysis completed!")

"""## Visualize AUC-PR Trend For Each Parameter"""

# Plot performance metrics for all parameters in one figure
print("\nAnalyzing coarse grid search results for parameters...\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Verify that results exist from previous cell
if 'train_sensitivity_results' not in globals() or 'val_sensitivity_results' not in globals():
    print("ERROR: Sensitivity results not found. Run the previous cell first.")
else:
    print(f"Found results for {len(train_sensitivity_results)} parameters")

    # Create a figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Get parameters to plot - all parameters with results
    plot_params = list(train_sensitivity_results.keys())
    print(f"Parameters to plot: {plot_params}")

    # Loop through parameters and create visualizations in each subplot
    for i, param_name in enumerate(plot_params):
        if i >= len(axes):  # Safety check in case there are more parameters than subplots
            break

        # Get the parameter values and corresponding performance metrics
        param_values_list = [val for val, _ in train_sensitivity_results[param_name]]
        train_scores = [score for _, score in train_sensitivity_results[param_name]]
        val_scores = [score for _, score in val_sensitivity_results[param_name]]

        # Find the best value first
        best_idx = val_scores.index(max(val_scores))
        best_value = param_values_list[best_idx]
        best_score = val_scores[best_idx]

        # Convert param values to strings for plotting if they contain None
        x_values = range(len(param_values_list))
        x_labels = [str(v) for v in param_values_list]

        # Get the current subplot
        ax1 = axes[i]

        # Plot training scores on primary y-axis
        color = 'blue'
        ax1.set_xlabel(f'Values of {param_name}')
        ax1.set_ylabel('Training AUC-PR', color=color)
        ax1.plot(x_values, train_scores, 'o-', color=color, linewidth=2, label='Training')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x_values)
        ax1.set_xticklabels(x_labels)

        # Create secondary y-axis for validation scores
        ax2 = ax1.twinx()
        color = 'green'
        ax2.set_ylabel('Validation AUC-PR', color=color)
        ax2.plot(x_values, val_scores, 's--', color=color, linewidth=2, label='Validation')
        ax2.tick_params(axis='y', labelcolor=color)

        # For n_estimators, adjust the validation axis scale
        if param_name == 'n_estimators':
            val_min = min(val_scores)
            val_max = max(val_scores)
            val_range = val_max - val_min
            buffer = max(val_range * 2, 0.001)  # Ensure visible variations
            ax2.set_ylim([max(0, val_min - buffer), min(1, val_max + buffer)])

        # Mark the best value
        ax2.scatter([best_idx], [best_score], color='red', s=100, zorder=5)
        ax2.axvline(x=best_idx, color='red', linestyle='--', alpha=0.3)

        # Add a title for this subplot
        ax1.set_title(f'Parameter: {param_name} (Best: {best_value})')

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Add grid
        ax1.grid(True, alpha=0.3)

        # Output the detailed performance metrics for this parameter
        print(f"\nPerformance metrics for {param_name}:")
        print(f"{'Value':<10} {'Training AUC-PR':<15} {'Validation AUC-PR':<15} {'Gap':<10}")
        print("-" * 50)

        for val, train_score, val_score in zip(param_values_list, train_scores, val_scores):
            gap = train_score - val_score
            # Mark the best value with an asterisk
            best_mark = " *" if val == best_value else ""
            print(f"{str(val):<10} {train_score:<15.4f} {val_score:<15.4f} {gap:<10.4f}{best_mark}")

    # Hide the unused subplots
    for i in range(len(plot_params), len(axes)):
        axes[i].axis('off')

    # Add a main title for the entire figure
    plt.suptitle('Parameter Sensitivity Analysis - Training and Validation AUC-PR', fontsize=16)

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle

    # Show the combined figure
    plt.show()

"""## Fine-grained Grid Search For Optimal Value of n_estimators"""

# Fine-Grained Grid Search for Random Forest with more robust AUC-PR scoring
print("Starting fine-grained grid search with robust AUC-PR scorer")

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import make_scorer
import warnings

# Define a more robust AUC-PR scorer function with detailed error handling
def auc_pr_score(y_true, y_prob):
    try:
        # Check for problematic conditions
        if len(np.unique(y_true)) < 2:
            warnings.warn(f"Only one class present in y_true. Returning 0.0")
            return 0.0

        # Some models might predict only one class
        if len(np.unique(y_prob)) < 2:
            warnings.warn(f"Constant probabilities detected. Returning 0.0")
            return 0.0

        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        # Calculate AUC with additional safeguards
        if len(precision) > 1 and len(recall) > 1:
            return auc(recall, precision)
        else:
            warnings.warn("Insufficient points for AUC calculation. Returning 0.0")
            return 0.0
    except Exception as e:
        warnings.warn(f"Error in AUC-PR calculation: {e}. Returning 0.0")
        return 0.0

# Create a custom scorer with debugging
auc_pr_scorer = make_scorer(auc_pr_score, needs_proba=True, greater_is_better=True)

# Define a more refined parameter grid
fine_param_grid = {
    'max_depth': [2],
    'min_samples_leaf': [8,9,10,11,12],
    'min_samples_split': [10,12,14,16,18,20,22,24,26,28,30],
    'n_estimators': [1],
    'max_features': [None],
    'random_state': [42]
}

# Run diagnostics on your data
print(f"y_train distribution: {y_train.value_counts()}")

# Let's try a direct cross-validation approach instead of GridSearchCV
from sklearn.model_selection import cross_val_score, KFold

# Initialize results dictionary
cv_results = {}

# Stratified K-Fold to maintain class balance
cv = StratifiedKFold(n_splits=100, shuffle=True, random_state=42)

print("Running direct cross-validation for each parameter combination...")
best_score = 0
best_params = None

# Iterate through parameter combinations
for min_samples_leaf in fine_param_grid['min_samples_leaf']:
    for min_samples_split in fine_param_grid['min_samples_split']:
        params = {
            'max_depth': 2,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'n_estimators': 1,
            'max_features': None,
            'random_state': 42
        }

        # Create model with these parameters
        model = RandomForestClassifier(**params)

        # Manual cross-validation with AUC-PR scoring
        fold_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled,y_train):
            # Split data
            X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train model
            model.fit(X_cv_train, y_cv_train)

            # Get probabilities
            y_cv_val_prob = model.predict_proba(X_cv_val)[:, 1]

            # Calculate AUC-PR manually
            try:
                precision, recall, _ = precision_recall_curve(y_cv_val, y_cv_val_prob)
                fold_auc_pr = auc(recall, precision)
                fold_scores.append(fold_auc_pr)
            except Exception as e:
                print(f"Error in fold for params {params}: {e}")
                fold_scores.append(0.0)

        # Average score across folds
        mean_score = np.mean(fold_scores)

        # Store result
        param_key = f"leaf={min_samples_leaf}, split={min_samples_split}"
        cv_results[param_key] = {
            'params': params,
            'mean_score': mean_score,
            'fold_scores': fold_scores
        }

        print(f"Params: {param_key} - Mean AUC-PR: {mean_score:.4f}, Fold scores: {fold_scores}")

        # Update best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

# Print best results
print(f"\nBest parameters: {best_params}")
print(f"Best AUC-PR score: {best_score:.4f}")

# Create best model for final evaluation
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train_scaled, y_train)

# Evaluate on training set
y_train_prob = best_model.predict_proba(X_train_scaled)[:, 1]
train_precision, train_recall, _ = precision_recall_curve(y_train, y_train_prob)
train_auc_pr = auc(train_recall, train_precision)

# Evaluate on validation set
y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]
val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_prob)
val_auc_pr = auc(val_recall, val_precision)

print(f"\nBest Model Evaluation:")
print(f"Training AUC-PR: {train_auc_pr:.4f}")
print(f"Validation AUC-PR: {val_auc_pr:.4f}")

# Plot precision-recall curves
plt.figure(figsize=(10, 6))
plt.plot(train_recall, train_precision, label=f'Training (AUC-PR = {train_auc_pr:.4f})')
plt.plot(val_recall, val_precision, label=f'Validation (AUC-PR = {val_auc_pr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Best Model')
plt.legend()
plt.grid(True)
plt.show()

"""## Visualize Fine-grained Grid Search Result"""

# Compare baseline model vs best fine-tuned model - focused on AUC-PR
print("Comparing baseline model and best fine-tuned model...")

# Import necessary metrics
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Define the baseline model (using default parameters)
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# Define the best fine-tuned model (already trained above)
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train_scaled, y_train)

# Calculate PR curves and AUC-PR values
# Baseline model
y_train_prob_baseline = baseline_model.predict_proba(X_train_scaled)[:, 1]
y_val_prob_baseline = baseline_model.predict_proba(X_val_scaled)[:, 1]

train_precision_baseline, train_recall_baseline, _ = precision_recall_curve(y_train, y_train_prob_baseline)
train_auc_pr_baseline = auc(train_recall_baseline, train_precision_baseline)

val_precision_baseline, val_recall_baseline, _ = precision_recall_curve(y_val, y_val_prob_baseline)
val_auc_pr_baseline = auc(val_recall_baseline, val_precision_baseline)

# Best model
y_train_prob_best = best_model.predict_proba(X_train_scaled)[:, 1]
y_val_prob_best = best_model.predict_proba(X_val_scaled)[:, 1]

train_precision_best, train_recall_best, _ = precision_recall_curve(y_train, y_train_prob_best)
train_auc_pr_best = auc(train_recall_best, train_precision_best)

val_precision_best, val_recall_best, _ = precision_recall_curve(y_val, y_val_prob_best)
val_auc_pr_best = auc(val_recall_best, val_precision_best)

# Calculate improvement percentage
improvement_absolute = val_auc_pr_best - val_auc_pr_baseline
improvement_percentage = (improvement_absolute / val_auc_pr_baseline) * 100

# Print AUC-PR values and improvement
print(f"\nBaseline model - Validation AUC-PR: {val_auc_pr_baseline:.4f}")
print(f"Best fine-tuned model - Validation AUC-PR: {val_auc_pr_best:.4f}")
print(f"Absolute improvement: {improvement_absolute:.4f}")
print(f"Percentage improvement: {improvement_percentage:.2f}%")

# Plot Precision-Recall curves for both models (validation set)
plt.figure(figsize=(10, 8))

# Plot PR curves
plt.plot(val_recall_baseline, val_precision_baseline,
         label=f'Baseline Model (AUC-PR = {val_auc_pr_baseline:.4f})',
         linestyle='--')
plt.plot(val_recall_best, val_precision_best,
         label=f'Best Fine-Tuned Model (AUC-PR = {val_auc_pr_best:.4f})')

# Add reference line for random classifier
plt.plot([0, 1], [y_val.mean(), y_val.mean()],
         linestyle=':', color='gray', label='Random Classifier')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison (Validation Set)')
plt.legend()
plt.grid(True)
plt.show()

# Print model parameters
print("\n==== Model Parameters ====")
print(f"Best Fine-Tuned Model: {best_model.get_params()}")

# ======================================================================
# Train model using the best parameters on combined training + validation data
# ======================================================================

print("Finding optimal threshold using combined training and validation sets...")

# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, precision_recall_curve, auc,
                             ConfusionMatrixDisplay)

# Combine training and validation sets
X_combined = np.vstack((X_train_scaled, X_val_scaled))
y_combined = pd.concat([y_train, y_val])

print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")

# Define best parameters
best_params = {
    'max_depth': 1,
    'min_samples_leaf': 4,
    'min_samples_split': 10,
    'n_estimators': 1,
    'random_state': 42
}

# Train model on combined dataset
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_combined, y_combined)

# Get probability predictions
y_combined_prob = best_model.predict_proba(X_combined)[:, 1]
y_combined_pred = best_model.predict(X_combined)  # Default threshold (0.5)

# Calculate AUC-PR and accuracy on combined dataset with default threshold
combined_precision, combined_recall, _ = precision_recall_curve(y_combined, y_combined_prob)
combined_auc_pr = auc(combined_recall, combined_precision)
combined_accuracy = accuracy_score(y_combined, y_combined_pred)
combined_weighted_precision = precision_score(y_combined, y_combined_pred, average='weighted')
combined_weighted_recall = recall_score(y_combined, y_combined_pred, average='weighted')
combined_weighted_f1 = f1_score(y_combined, y_combined_pred, average='weighted')

print(f"Combined dataset metrics (default threshold 0.5):")
print(f"AUC-PR: {combined_auc_pr:.4f}")
print(f"Accuracy: {combined_accuracy:.4f}")
print(f"Weighted Precision: {combined_weighted_precision:.4f}")
print(f"Weighted Recall: {combined_weighted_recall:.4f}")
print(f"Weighted F1 Score: {combined_weighted_f1:.4f}")

# ======================================================================
# Use a tested optimal threshold of 0.12
# ======================================================================

fixed_threshold = 0.12
print(f"\nUsing a fixed threshold of {fixed_threshold} for predictions...")

# Convert probabilities to binary predictions using the fixed threshold for the combined set
y_combined_pred_fixed = (y_combined_prob >= fixed_threshold).astype(int)

# Calculate metrics at the fixed threshold (combined data)
combined_fixed_precision = precision_score(y_combined, y_combined_pred_fixed, average='weighted')
combined_fixed_recall = recall_score(y_combined, y_combined_pred_fixed, average='weighted')
combined_fixed_f1 = f1_score(y_combined, y_combined_pred_fixed, average='weighted')
combined_fixed_accuracy = accuracy_score(y_combined, y_combined_pred_fixed)

# Generate confusion matrix for combined set
cm_combined = confusion_matrix(y_combined, y_combined_pred_fixed)

print("\n=== Combined dataset metrics (threshold=0.1) ===")
print(f"AUC-PR: {combined_auc_pr:.4f}")  # AUC-PR remains the same, just reusing the probabilities
print(f"Precision: {combined_fixed_precision:.4f}")
print(f"Recall: {combined_fixed_recall:.4f}")
print(f"F1 Score: {combined_fixed_f1:.4f}")
print(f"Accuracy: {combined_fixed_accuracy:.4f}")

# ======================================================================
# Evaluate model on test set using fixed threshold
# ======================================================================

print("\nEvaluating model on test set using threshold=0.1...")

y_test = test_data['label']
y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred_fixed = (y_test_prob >= fixed_threshold).astype(int)
y_test_pred_default = best_model.predict(X_test_scaled)  # Default threshold (0.5)

# Calculate test metrics with fixed threshold
test_fixed_precision = precision_score(y_test, y_test_pred_fixed, average='weighted')
test_fixed_recall = recall_score(y_test, y_test_pred_fixed, average='weighted')
test_fixed_f1 = f1_score(y_test, y_test_pred_fixed, average='weighted')
test_fixed_accuracy = accuracy_score(y_test, y_test_pred_fixed)

# Generate confusion matrix for test set
cm_test = confusion_matrix(y_test, y_test_pred_fixed)

# Calculate AUC-PR on test set
test_precision_curve, test_recall_curve, _ = precision_recall_curve(y_test, y_test_prob)
test_auc_pr = auc(test_recall_curve, test_precision_curve)

print("\n=== Test set metrics (threshold=0.1) ===")
print(f"AUC-PR: {test_auc_pr:.4f}")
print(f"Precision: {test_fixed_precision:.4f}")
print(f"Recall: {test_fixed_recall:.4f}")
print(f"F1 Score: {test_fixed_f1:.4f}")
print(f"Accuracy: {test_fixed_accuracy:.4f}")

# ======================================================================
# Create new performance comparison chart (threshold=0.1)
# ======================================================================

comparison_data_fixed = {
    'Metric': ['AUC-PR', 'Weighted F1 Score', 'Weighted Precision', 'Weighted Recall', 'Accuracy'],
    'Combined Training (0.1)': [
        combined_auc_pr,    # AUC-PR for combined set
        combined_fixed_f1,
        combined_fixed_precision,
        combined_fixed_recall,
        combined_fixed_accuracy
    ],
    'Test Set (0.1)': [
        test_auc_pr,
        test_fixed_f1,
        test_fixed_precision,
        test_fixed_recall,
        test_fixed_accuracy
    ]
}

comparison_df_fixed = pd.DataFrame(comparison_data_fixed)
print("\n==== Training vs Test Performance Comparison (threshold=0.1) ====")
pd.set_option('display.float_format', '{:.4f}'.format)
print(comparison_df_fixed)

plt.figure(figsize=(12, 6))
x = np.arange(len(comparison_data_fixed['Metric']))
width = 0.35

plt.bar(x - width/2, comparison_data_fixed['Combined Training (0.1)'], width, label='Combined', color='skyblue')
plt.bar(x + width/2, comparison_data_fixed['Test Set (0.1)'], width, label='Test', color='lightgreen')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Training vs Test Performance (threshold=0.1)')
plt.xticks(x, comparison_data_fixed['Metric'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(comparison_data_fixed['Combined Training (0.1)']):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
for i, v in enumerate(comparison_data_fixed['Test Set (0.1)']):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.show()

# ======================================================================
# Side-by-side confusion matrices for combined and test sets
# ======================================================================

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(cm_combined).plot(ax=axs[0], cmap='Blues')
axs[0].set_title('Confusion Matrix - Combined (threshold=0.1)')

ConfusionMatrixDisplay(cm_test).plot(ax=axs[1], cmap='Greens')
axs[1].set_title('Confusion Matrix - Test (threshold=0.1)')

plt.tight_layout()
plt.show()

# ======================================================================
# Final model summary
# ======================================================================

print("\n======= Final Model Summary =======")
print("Best hyperparameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"Fixed threshold: {fixed_threshold}")
print(f"Test set AUC-PR: {test_auc_pr:.4f}")
print(f"Test set Accuracy: {test_fixed_accuracy:.4f}")
print(f"Test set Weighted F1-Score: {test_fixed_f1:.4f}")

# ======================================================================
# Add predictions to the original test data and export to CSV
# ======================================================================

print("\nAdding predictions to original test data and exporting to CSV...")

try:
    original_test_df = test_data.copy()
except NameError:
    print("Original test dataframe not found. Please ensure test_df is defined or load from file.")

# Add prediction columns to the original test dataframe
original_test_df['predicted_probability'] = y_test_prob
original_test_df['predicted_label'] = y_test_pred_fixed  # Using fixed threshold=0.1

# Export the augmented test data to CSV
output_filename = 'test_with_predictions_fixed_threshold.csv'
original_test_df.to_csv(output_filename, index=False)
print(f"Exported predictions to {output_filename}")

# Display a sample of the predictions
print("\nSample of test data with predictions:")
pd.set_option('display.max_columns', 10)
print(original_test_df.head())