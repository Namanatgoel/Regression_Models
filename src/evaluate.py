import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (mean_squared_error, r2_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, 
                             average_precision_score)

def plot_correlation_heatmap(df: pd.DataFrame, model_name: str):
    """
    Computes and saves the Pearson correlation matrix heatmap.
    Used to diagnose multicollinearity prior to model training.
    """
    path = f'outputs/{model_name.lower()}'
    os.makedirs(path, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    # Isolate numeric metrics to prevent casting errors
    numeric_df = df.select_dtypes(include=[np.number])
    
    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu_r', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Feature Correlation Matrix ({model_name} Pipeline)')
    plt.tight_layout()
    plt.savefig(f'{path}/correlation_heatmap.png')
    plt.close()
    print(f"Metrics Log -> Correlation heatmap saved to {path}/correlation_heatmap.png")

def evaluate_linear(y_true: np.ndarray, y_pred: np.ndarray, feature_names: list, weights: np.ndarray):
    """Generates all Linear Regression visualizations and logs metrics."""
    os.makedirs('outputs/linear', exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', color='teal')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted (Homoscedasticity Check)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/linear/residuals.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', color='navy')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual MEDV')
    plt.ylabel('Predicted MEDV')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/linear/actual_vs_pred.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    coeffs = pd.Series(weights, index=feature_names).sort_values()
    coeffs.plot(kind='barh', color='crimson')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Model Coefficients)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/linear/feature_importance.png')
    plt.close()
    
    print(f"Linear Metrics -> MSE: {mean_squared_error(y_true, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}, R2: {r2_score(y_true, y_pred):.4f}")

def evaluate_logistic(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Generates all Logistic Regression visualizations and logs metrics."""
    os.makedirs('outputs/logistic', exist_ok=True)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/logistic/confusion_matrix.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.4f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/logistic/roc_curve.png')
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'AP (PR-AUC) = {ap_score:.4f}', color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Imbalanced Class Metric)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/logistic/pr_curve.png')
    plt.close()
    
    print(f"Logistic Metrics -> Precision: {precision_score(y_true, y_pred):.4f}, Recall: {recall_score(y_true, y_pred):.4f}, F1: {f1_score(y_true, y_pred):.4f}, AUC: {roc_auc_score(y_true, y_prob):.4f}")