"""
Ablation Study: Component-wise Analysis
========================================

ORIGINAL CONTRIBUTION for Graduation Project

This script performs a comprehensive ablation study to understand
the contribution of each component in our ticket classification system.

Ablation Study = Removing components one-by-one to measure their impact

This is a KEY requirement for academic projects to show:
- What works and what doesn't
- Why your approach is better
- Scientific rigor

Author: [Your Name]
Date: November 2025
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('..')
from utils import basic_clean
from custom_features import ITTicketFeatureExtractor
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and split data"""
    df = pd.read_csv("../data/cleaned_data.csv")
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_tfidf_only():
    """
    Experiment 1: TF-IDF only (baseline)
    No deep learning, no custom features
    """
    print("\n[Experiment 1] TF-IDF + Logistic Regression (Baseline)")
    print("-" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)
    
    pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'name': 'TF-IDF Only',
        'accuracy': acc,
        'f1_score': f1,
        'improvement': 0.0
    }


def evaluate_lstm_basic():
    """
    Experiment 2: Word2Vec + Basic LSTM
    Deep learning but no custom attention
    """
    print("\n[Experiment 2] Word2Vec + Basic LSTM")
    print("-" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Load models
    lstm_model = load_model("../models/word2vec_lstm_model.h5")
    with open("../models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("../models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Preprocess
    X_test_clean = X_test.apply(basic_clean)
    sequences = tokenizer.texts_to_sequences(X_test_clean)
    padded = pad_sequences(sequences, maxlen=80, padding="post", truncating="post")
    
    # Predict
    y_test_encoded = label_encoder.transform(y_test)
    pred = lstm_model.predict(padded, verbose=0).argmax(axis=1)
    
    acc = accuracy_score(y_test_encoded, pred)
    f1 = f1_score(y_test_encoded, pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'name': 'Word2Vec + LSTM',
        'accuracy': acc,
        'f1_score': f1
    }


def evaluate_with_custom_features():
    """
    Experiment 3: TF-IDF + Custom Domain Features
    Shows impact of domain knowledge
    """
    print("\n[Experiment 3] TF-IDF + Custom Domain Features")
    print("-" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Custom features
    feature_extractor = ITTicketFeatureExtractor()
    custom_train = feature_extractor.extract_batch(X_train)
    custom_test = feature_extractor.extract_batch(X_test)
    
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, custom_train.values])
    X_test_combined = hstack([X_test_tfidf, custom_test.values])
    
    # Train
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X_train_combined, y_train)
    
    pred = clf.predict(X_test_combined)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'name': 'TF-IDF + Custom Features',
        'accuracy': acc,
        'f1_score': f1
    }


def evaluate_ensemble():
    """
    Experiment 4: Ensemble (Baseline + LSTM)
    Shows benefit of model combination
    """
    print("\n[Experiment 4] Ensemble (Baseline + LSTM)")
    print("-" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Load models
    with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    
    lstm_model = load_model("../models/word2vec_lstm_model.h5")
    with open("../models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("../models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Baseline predictions
    baseline_probs = baseline_model.predict_proba(X_test)
    
    # LSTM predictions
    X_test_clean = X_test.apply(basic_clean)
    sequences = tokenizer.texts_to_sequences(X_test_clean)
    padded = pad_sequences(sequences, maxlen=80, padding="post", truncating="post")
    lstm_probs = lstm_model.predict(padded, verbose=0)
    
    # Ensemble (equal weights)
    ensemble_probs = (baseline_probs + lstm_probs) / 2
    pred = ensemble_probs.argmax(axis=1)
    pred_labels = label_encoder.inverse_transform(pred)
    
    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average='weighted')
    
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'name': 'Ensemble (Baseline + LSTM)',
        'accuracy': acc,
        'f1_score': f1
    }


def run_ablation_study():
    """
    Run complete ablation study
    
    This systematically evaluates each component's contribution
    """
    print("=" * 70)
    print("ABLATION STUDY - SYSTEMATIC COMPONENT ANALYSIS")
    print("=" * 70)
    print("\nThis study evaluates the contribution of each component:")
    print("  1. Baseline features (TF-IDF)")
    print("  2. Deep learning (Word2Vec + LSTM)")
    print("  3. Domain knowledge (Custom features)")
    print("  4. Model combination (Ensemble)")
    print("\nRunning experiments...\n")
    
    results = []
    
    # Experiment 1: Baseline
    baseline_result = evaluate_tfidf_only()
    results.append(baseline_result)
    baseline_acc = baseline_result['accuracy']
    
    # Experiment 2: LSTM
    lstm_result = evaluate_lstm_basic()
    lstm_result['improvement'] = (lstm_result['accuracy'] - baseline_acc) * 100
    results.append(lstm_result)
    
    # Experiment 3: Custom Features
    custom_result = evaluate_with_custom_features()
    custom_result['improvement'] = (custom_result['accuracy'] - baseline_acc) * 100
    results.append(custom_result)
    
    # Experiment 4: Ensemble
    ensemble_result = evaluate_ensemble()
    ensemble_result['improvement'] = (ensemble_result['accuracy'] - baseline_acc) * 100
    results.append(ensemble_result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv("../reports/ablation_study_results.csv", index=False)
    print("\n✅ Results saved to: reports/ablation_study_results.csv")
    
    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    best = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\n🏆 Best Approach: {best['name']}")
    print(f"   Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"   Improvement over baseline: +{best['improvement']:.2f}%")
    
    print("\n📊 Component Contributions:")
    for _, row in results_df.iterrows():
        if row['improvement'] > 0:
            print(f"   • {row['name']}: +{row['improvement']:.2f}%")
    
    print("\n💡 Insights:")
    if lstm_result['improvement'] > 0:
        print(f"   ✓ Deep learning provides +{lstm_result['improvement']:.2f}% improvement")
    if custom_result['improvement'] > 0:
        print(f"   ✓ Domain features provide +{custom_result['improvement']:.2f}% improvement")
    if ensemble_result['improvement'] > 0:
        print(f"   ✓ Ensemble provides +{ensemble_result['improvement']:.2f}% improvement")
    
    print("\n" + "=" * 70)
    print("ACADEMIC VALUE")
    print("=" * 70)
    print("""
This ablation study demonstrates:
  1. Scientific rigor (systematic evaluation)
  2. Understanding of each component
  3. Justification for design choices
  4. Empirical validation of approach

This is a REQUIRED component for academic projects!
    """)
    
    return results_df


if __name__ == "__main__":
    try:
        results = run_ablation_study()
        print("\n✅ Ablation study completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure all models are trained before running ablation study.")
        print("Run these first:")
        print("  1. src/01_baseline_tfidf_logreg.ipynb")
        print("  2. src/02_word2vec_lstm.ipynb")

