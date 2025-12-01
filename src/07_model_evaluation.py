"""
Comprehensive Model Evaluation & Comparison
Confusion Matrix, ROC/PR Curves, Error Analysis & Benchmark Table
"""

from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append("..")
from utils import basic_clean  # noqa: E402

REPORT_DIR = Path("../reports")
DATA_PATH = Path("../data/cleaned_data.csv")
MAX_LEN = 80


def ensure_report_dir():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def split_dataset():
    df = pd.read_csv(DATA_PATH)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=42,
        stratify=y_tmp,
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def plot_confusion_matrix(y_true, y_pred, class_labels, title, save_path):
    """Plot confusion matrix with ordered axes."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(f"Confusion Matrix · {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Confusion matrix saved → {save_path}")


def _binarize_labels(y_true, class_labels):
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    indices = np.array([label_to_idx[label] for label in y_true])
    return label_binarize(indices, classes=range(len(class_labels)))


def plot_roc_curves(y_true, y_pred_proba, class_labels, title, save_path):
    """Plot multi-class ROC curves."""
    y_true_bin = _binarize_labels(y_true, class_labels)
    n_classes = len(class_labels)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{class_labels[i]} · AUC={roc_auc[i]:.2f}",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves · {title}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] ROC curves saved → {save_path}")


def plot_pr_curves(y_true, y_pred_proba, class_labels, title, save_path):
    """Plot precision-recall curves for each class."""
    y_true_bin = _binarize_labels(y_true, class_labels)
    n_classes = len(class_labels)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        avg_precision = average_precision_score(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        plt.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{class_labels[i]} · AP={avg_precision:.2f}",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves · {title}")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] PR curves saved → {save_path}")


def error_analysis(texts, y_true, y_pred, save_path, top_k=10):
    """Save and print most common misclassifications."""
    mask = y_true != y_pred
    error_df = pd.DataFrame(
        {
            "text": texts[mask],
            "true_label": y_true[mask],
            "predicted_label": y_pred[mask],
        }
    )
    error_df.to_csv(save_path, index=False)

    print("\n=== ERROR ANALYSIS ===")
    total_errors = len(error_df)
    total_samples = len(y_true)
    print(f"Total errors: {total_errors}")
    print(f"Error rate : {total_errors / total_samples:.2%}")
    confusion_pairs = (
        error_df.groupby(["true_label", "predicted_label"])
        .size()
        .sort_values(ascending=False)
        .head(top_k)
    )
    print("\nMost confused pairs:")
    print(confusion_pairs)
    print(f"[✓] Error samples saved → {save_path}")
    return error_df


def create_comparison_table(models_results, save_path):
    """Create comparison table (CSV + bar plot)."""
    comparison_df = pd.DataFrame(models_results)
    comparison_df.to_csv(save_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        kind="bar",
        ax=ax,
        rot=25,
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim([0.7, 1.0])
    plt.legend(loc="lower right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=300)
    plt.close()

    print(f"[✓] Model comparison saved → {save_path}")
    print(comparison_df)


def evaluate_baseline(X_test, y_test):
    with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
        baseline_model = pickle.load(f)

    preds = baseline_model.predict(X_test)
    probs = baseline_model.predict_proba(X_test)
    classes = baseline_model.classes_.tolist()

    print("\n=== BASELINE MODEL (TF-IDF + Logistic Regression) ===")
    print(classification_report(y_test, preds, zero_division=0))

    metrics = {
        "Model": "Baseline (TF-IDF + LogReg)",
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(
            y_test, preds, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, preds, average="weighted", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, preds, average="weighted", zero_division=0
        ),
    }

    return {
        "preds": preds,
        "probs": probs,
        "classes": classes,
        "metrics": metrics,
    }


def evaluate_lstm(X_test, y_test):
    lstm_model = load_model("../models/word2vec_lstm_model.h5")
    with open("../models/tokenizer.pkl", "rb") as f_tok:
        tokenizer = pickle.load(f_tok)
    with open("../models/label_encoder.pkl", "rb") as f_enc:
        label_encoder = pickle.load(f_enc)

    cleaned = X_test.apply(basic_clean)
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(
        sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    probs = lstm_model.predict(padded, verbose=0)
    pred_indices = probs.argmax(axis=1)
    preds = label_encoder.inverse_transform(pred_indices)
    classes = label_encoder.classes_.tolist()

    print("\n=== LSTM MODEL (Word2Vec + BiLSTM) ===")
    print(classification_report(y_test, preds, zero_division=0))

    metrics = {
        "Model": "Word2Vec + BiLSTM",
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(
            y_test, preds, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, preds, average="weighted", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, preds, average="weighted", zero_division=0
        ),
    }

    return {
        "preds": preds,
        "probs": probs,
        "classes": classes,
        "metrics": metrics,
    }


def main():
    ensure_report_dir()
    _, _, X_test, _, _, y_test = split_dataset()

    baseline_eval = evaluate_baseline(X_test, y_test)
    lstm_eval = evaluate_lstm(X_test, y_test)

    baseline_preds = pd.Series(baseline_eval["preds"], index=y_test.index)
    lstm_preds = pd.Series(lstm_eval["preds"], index=y_test.index)

    # Baseline visuals
    plot_confusion_matrix(
        y_test,
        baseline_preds,
        baseline_eval["classes"],
        "Baseline (TF-IDF + LogReg)",
        REPORT_DIR / "baseline_confusion_matrix.png",
    )
    plot_roc_curves(
        y_test,
        baseline_eval["probs"],
        baseline_eval["classes"],
        "Baseline (TF-IDF + LogReg)",
        REPORT_DIR / "baseline_roc_curves.png",
    )
    plot_pr_curves(
        y_test,
        baseline_eval["probs"],
        baseline_eval["classes"],
        "Baseline (TF-IDF + LogReg)",
        REPORT_DIR / "baseline_pr_curves.png",
    )
    error_analysis(
        X_test,
        y_test,
        baseline_preds,
        REPORT_DIR / "baseline_error_samples.csv",
    )

    # LSTM visuals
    plot_confusion_matrix(
        y_test,
        lstm_preds,
        lstm_eval["classes"],
        "Word2Vec + BiLSTM",
        REPORT_DIR / "lstm_confusion_matrix.png",
    )
    plot_roc_curves(
        y_test,
        lstm_eval["probs"],
        lstm_eval["classes"],
        "Word2Vec + BiLSTM",
        REPORT_DIR / "lstm_roc_curves.png",
    )
    plot_pr_curves(
        y_test,
        lstm_eval["probs"],
        lstm_eval["classes"],
        "Word2Vec + BiLSTM",
        REPORT_DIR / "lstm_pr_curves.png",
    )
    error_analysis(
        X_test,
        y_test,
        lstm_preds,
        REPORT_DIR / "lstm_error_samples.csv",
    )

    # Aggregated comparison
    models_results = [baseline_eval["metrics"], lstm_eval["metrics"]]
    create_comparison_table(
        models_results, REPORT_DIR / "model_comparison_detailed.csv"
    )

    print("\n✅ Comprehensive evaluation artefacts saved under reports/.")


if __name__ == "__main__":
    main()

