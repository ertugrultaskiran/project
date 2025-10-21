"""
Mevcut Modellerin Detaylı Değerlendirmesi
Tüm modelleri test edip karşılaştırır
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('..')
from utils import basic_clean

print("=" * 60)
print("MODEL DEĞERLENDIRME BAŞLIYOR")
print("=" * 60)

# 1. VERİYİ YÜKLE
print("\n[1/5] Veri yükleniyor...")
df = pd.read_csv("../data/cleaned_data.csv")
print(f"[OK] Veri yuklendi: {df.shape}")

# 2. TRAIN/TEST SPLIT (aynı parametrelerle)
print("\n[2/5] Veri bölünüyor...")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)
print(f"[OK] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 3. BASELINE MODEL DEĞERLENDİR
print("\n[3/5] Baseline model değerlendiriliyor...")
with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
    baseline_model = pickle.load(f)

baseline_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

print(f"\n{'='*60}")
print("BASELINE MODEL (TF-IDF + Logistic Regression)")
print(f"{'='*60}")
print(f"Test Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print("\nDetaylı Rapor:")
print(classification_report(y_test, baseline_pred, zero_division=0))

# 4. LSTM MODEL DEĞERLENDİR
print("\n[4/5] LSTM model değerlendiriliyor...")
lstm_model = load_model("../models/word2vec_lstm_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Preprocess for LSTM
X_test_clean = X_test.apply(basic_clean)
sequences = tokenizer.texts_to_sequences(X_test_clean)
padded = pad_sequences(sequences, maxlen=80, padding="post", truncating="post")

# Predict
lstm_probs = lstm_model.predict(padded, verbose=0)
lstm_pred_idx = lstm_probs.argmax(axis=1)
lstm_pred = label_encoder.inverse_transform(lstm_pred_idx)

# Encode y_test for comparison
y_test_encoded = label_encoder.transform(y_test)
lstm_acc = accuracy_score(y_test_encoded, lstm_pred_idx)

print(f"\n{'='*60}")
print("LSTM MODEL (Word2Vec + Bidirectional LSTM)")
print(f"{'='*60}")
print(f"Test Accuracy: {lstm_acc:.4f} ({lstm_acc*100:.2f}%)")
print("\nDetaylı Rapor:")
print(classification_report(y_test, lstm_pred, zero_division=0))

# 5. KARŞILAŞTIRMA
print("\n[5/5] Model karşılaştırması...")
print(f"\n{'='*60}")
print("MODEL KARŞILAŞTIRMASI")
print(f"{'='*60}")
print(f"Baseline (TF-IDF + LogReg): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"LSTM (Word2Vec + BiLSTM):   {lstm_acc:.4f} ({lstm_acc*100:.2f}%)")
print(f"İyileştirme:                 +{(lstm_acc-baseline_acc)*100:.2f}%")

# 6. CONFUSION MATRIX GÖRSELLEŞTIR
print("\n[6/6] Confusion matrix oluşturuluyor...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Baseline confusion matrix
cm_baseline = confusion_matrix(y_test, baseline_pred)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(df['label'].unique()),
            yticklabels=sorted(df['label'].unique()),
            ax=axes[0])
axes[0].set_title(f'Baseline Model - Accuracy: {baseline_acc:.2%}')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# LSTM confusion matrix
cm_lstm = confusion_matrix(y_test, lstm_pred)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens',
            xticklabels=sorted(df['label'].unique()),
            yticklabels=sorted(df['label'].unique()),
            ax=axes[1])
axes[1].set_title(f'LSTM Model - Accuracy: {lstm_acc:.2%}')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../reports/model_comparison_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[OK] Confusion matrix kaydedildi: reports/model_comparison_confusion_matrix.png")

# 7. PER-CLASS KARŞILAŞTIRMA
print("\n[7/7] Sınıf bazında performans karşılaştırması...")

from sklearn.metrics import precision_recall_fscore_support

# Baseline metrics
p_base, r_base, f1_base, _ = precision_recall_fscore_support(y_test, baseline_pred, average=None, zero_division=0)

# LSTM metrics
p_lstm, r_lstm, f1_lstm, _ = precision_recall_fscore_support(y_test, lstm_pred, average=None, zero_division=0)

classes = sorted(df['label'].unique())
comparison_df = pd.DataFrame({
    'Class': classes,
    'Baseline_Precision': p_base,
    'LSTM_Precision': p_lstm,
    'Baseline_Recall': r_base,
    'LSTM_Recall': r_lstm,
    'Baseline_F1': f1_base,
    'LSTM_F1': f1_lstm
})

print("\nSınıf Bazında Karşılaştırma:")
print(comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('../reports/per_class_comparison.csv', index=False)
print("\n[OK] Detayli karsilastirma kaydedildi: reports/per_class_comparison.csv")

# 8. ÖNERİLER
print(f"\n{'='*60}")
print("İYİLEŞTİRME ÖNERİLERİ")
print(f"{'='*60}")

# En kötü performans gösteren sınıflar
worst_classes = comparison_df.nsmallest(3, 'LSTM_F1')
print("\nEn düşük F1-score'a sahip sınıflar (LSTM):")
for _, row in worst_classes.iterrows():
    print(f"  - {row['Class']}: F1={row['LSTM_F1']:.3f}")
    
print("\nÖneriler:")
print("  1. Data augmentation uygulayın (özellikle düşük performanslı sınıflara)")
print("  2. Hyperparameter tuning yapın (dropout, learning rate, LSTM units)")
print("  3. BERT fine-tuning deneyin (%90+ accuracy için)")
print("  4. Ensemble modeling yapın (baseline + LSTM)")

print(f"\n{'='*60}")
print("DEĞERLENDİRME TAMAMLANDI!")
print(f"{'='*60}")
print("\nOluşturulan dosyalar:")
print("  - reports/model_comparison_confusion_matrix.png")
print("  - reports/per_class_comparison.csv")

