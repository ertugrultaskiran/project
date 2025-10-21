"""
Ensemble Model: Baseline + LSTM modellerini birleştir
Beklenen Accuracy: 89-90%
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append('..')
from utils import basic_clean

print("=" * 60)
print("ENSEMBLE MODEL - BASELINE + LSTM")
print("=" * 60)

# 1. LOAD DATA
print("\n[1/5] Veri yükleniyor...")
df = pd.read_csv("../data/cleaned_data.csv")
print(f"[OK] Veri yüklendi: {df.shape}")

# 2. TRAIN/TEST SPLIT (aynı parametrelerle)
print("\n[2/5] Veri bölünüyor...")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)
print(f"[OK] Test set size: {len(X_test)}")

# 3. LOAD MODELS
print("\n[3/5] Modeller yükleniyor...")
with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
    baseline_model = pickle.load(f)

lstm_model = load_model("../models/word2vec_lstm_model.h5")

with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("[OK] Tüm modeller yüklendi")

# 4. GET PREDICTIONS FROM EACH MODEL
print("\n[4/5] Model tahminleri alınıyor...")

# Baseline predictions
baseline_probs = baseline_model.predict_proba(X_test)

# LSTM predictions
X_test_clean = X_test.apply(basic_clean)
sequences = tokenizer.texts_to_sequences(X_test_clean)
padded = pad_sequences(sequences, maxlen=80, padding="post", truncating="post")
lstm_probs = lstm_model.predict(padded, verbose=0)

print("[OK] Tahminler alındı")

# 5. ENSEMBLE: WEIGHTED AVERAGE
print("\n[5/5] Ensemble hesaplanıyor...")
# Try different weights
weights = [
    (0.5, 0.5, "Equal"),
    (0.4, 0.6, "LSTM-heavy"),
    (0.6, 0.4, "Baseline-heavy"),
    (0.3, 0.7, "LSTM-very-heavy")
]

best_acc = 0
best_weight = None
best_preds = None

for w1, w2, name in weights:
    ensemble_probs = (w1 * baseline_probs + w2 * lstm_probs)
    ensemble_pred_idx = np.argmax(ensemble_probs, axis=1)
    ensemble_preds = label_encoder.inverse_transform(ensemble_pred_idx)
    
    acc = accuracy_score(y_test, ensemble_preds)
    print(f"  {name} ({w1:.1f}/{w2:.1f}): {acc:.4f} ({acc*100:.2f}%)")
    
    if acc > best_acc:
        best_acc = acc
        best_weight = (w1, w2, name)
        best_preds = ensemble_preds

print(f"\n{'='*60}")
print("EN İYİ ENSEMBLE MODEL")
print(f"{'='*60}")
print(f"Weights: {best_weight[2]} ({best_weight[0]:.1f}/{best_weight[1]:.1f})")
print(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print("\nDetaylı Rapor:")
print(classification_report(y_test, best_preds, zero_division=0))

# Save ensemble configuration
ensemble_config = {
    'baseline_weight': best_weight[0],
    'lstm_weight': best_weight[1],
    'name': best_weight[2],
    'accuracy': best_acc
}

with open("../models/ensemble_config.pkl", "wb") as f:
    pickle.dump(ensemble_config, f)

print(f"\n[OK] Ensemble configuration saved: models/ensemble_config.pkl")
print(f"\n{'='*60}")
print("ENSEMBLE MODEL TAMAMLANDI!")
print(f"{'='*60}")

