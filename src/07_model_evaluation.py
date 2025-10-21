"""
Comprehensive Model Evaluation & Comparison
Confusion Matrix, ROC Curves, PR Curves, Error Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import pickle

# Load test data
df = pd.read_csv("../data/cleaned_data.csv")
# ... (split data)

# Load models
with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
    baseline_model = pickle.load(f)

# 1. CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")

# 2. ROC CURVES (Multi-class)
def plot_roc_curves(y_true, y_pred_proba, classes, title, save_path):
    """Plot ROC curves for all classes"""
    n_classes = len(classes)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {title}')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curves saved: {save_path}")

# 3. PRECISION-RECALL CURVES
def plot_pr_curves(y_true, y_pred_proba, classes, title, save_path):
    """Plot Precision-Recall curves"""
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                label=f'{classes[i]} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {title}')
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PR curves saved: {save_path}")

# 4. ERROR ANALYSIS
def error_analysis(X_test, y_true, y_pred, classes, save_path):
    """Analyze misclassified examples"""
    errors = X_test[y_true != y_pred]
    true_labels = y_true[y_true != y_pred]
    pred_labels = y_pred[y_true != y_pred]
    
    error_df = pd.DataFrame({
        'text': errors,
        'true_label': [classes[i] for i in true_labels],
        'predicted_label': [classes[i] for i in pred_labels]
    })
    
    # Save errors
    error_df.to_csv(save_path, index=False)
    
    # Print statistics
    print(f"\n=== ERROR ANALYSIS ===")
    print(f"Total errors: {len(error_df)}")
    print(f"Error rate: {len(error_df) / len(y_true):.2%}")
    print(f"\nMost confused pairs:")
    print(error_df.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False).head(10))
    
    return error_df

# 5. MODEL COMPARISON TABLE
def create_comparison_table(models_results, save_path):
    """Create comparison table of all models"""
    comparison_df = pd.DataFrame(models_results)
    
    # Save as CSV
    comparison_df.to_csv(save_path, index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                       kind='bar', ax=ax, rot=45)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim([0.7, 1.0])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path.replace('.csv', '.png'), dpi=300)
    plt.close()
    
    print(f"\nModel comparison saved: {save_path}")
    print(comparison_df)

if __name__ == "__main__":
    print("Running comprehensive model evaluation...")
    # Run all evaluation functions
    # plot_confusion_matrix(...)
    # plot_roc_curves(...)
    # plot_pr_curves(...)
    # error_analysis(...)
    # create_comparison_table(...)

