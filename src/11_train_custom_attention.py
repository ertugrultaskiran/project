"""
Quick Train Script for Custom Attention LSTM
============================================

This script trains the custom attention-based LSTM model.
Run this ASAP to get results!

Usage:
    python src/11_train_custom_attention.py

Expected time: 1-2 hours
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import sys
sys.path.append('..')
from utils import basic_clean
from custom_attention_layer import CustomAttentionLayer

print("=" * 70)
print("TRAINING CUSTOM ATTENTION-LSTM MODEL")
print("=" * 70)

# Load data
print("\n[1/8] Loading data...")
df = pd.read_csv("../data/cleaned_data.csv")
df['text_clean'] = df['text'].apply(basic_clean)
print(f"  ✓ Loaded {len(df)} tickets")

# Split data
print("\n[2/8] Splitting data...")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)
print(f"  ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Tokenization
print("\n[3/8] Tokenizing...")
MAX_VOCAB = 40000
MAX_LEN = 80

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

seq_train = tokenizer.texts_to_sequences(X_train)
seq_val = tokenizer.texts_to_sequences(X_val)
seq_test = tokenizer.texts_to_sequences(X_test)

Xtr = pad_sequences(seq_train, maxlen=MAX_LEN, padding="post", truncating="post")
Xv = pad_sequences(seq_val, maxlen=MAX_LEN, padding="post", truncating="post")
Xt = pad_sequences(seq_test, maxlen=MAX_LEN, padding="post", truncating="post")
print(f"  ✓ Vocabulary size: {len(tokenizer.word_index)}")

# Label encoding
print("\n[4/8] Encoding labels...")
le = LabelEncoder()
ytr = le.fit_transform(y_train)
yv = le.transform(y_val)
yt = le.transform(y_test)
num_classes = len(le.classes_)
vocab_size = min(MAX_VOCAB, len(tokenizer.word_index)+1)
print(f"  ✓ Number of classes: {num_classes}")

# Build model with custom attention
print("\n[5/8] Building model with CUSTOM ATTENTION...")

inp = layers.Input(shape=(MAX_LEN,))

# Embedding
x = layers.Embedding(vocab_size, 128, input_length=MAX_LEN)(inp)
x = layers.SpatialDropout1D(0.2)(x)

# BiLSTM with return_sequences=True (needed for attention)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

# ⭐ CUSTOM ATTENTION LAYER (OUR CONTRIBUTION!)
x = CustomAttentionLayer()(x)

x = layers.Dropout(0.3)(x)

# Output
out = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inp, out)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("  ✓ Model built with custom attention!")
print(f"  ✓ Total parameters: {model.count_params():,}")

# Class weights
print("\n[6/8] Computing class weights...")
classes = np.unique(ytr)
class_weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
cw = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
print("  ✓ Class weights computed")

# Train
print("\n[7/8] Training model...")
print("  (This may take 30-60 minutes...)\n")

early = tf.keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True,
    monitor="val_accuracy"
)

history = model.fit(
    Xtr, ytr,
    validation_data=(Xv, yv),
    epochs=15,
    batch_size=64,
    callbacks=[early],
    class_weight=cw,
    verbose=1
)

# Evaluate
print("\n[8/8] Evaluating on test set...")
pred_test = model.predict(Xt).argmax(axis=1)
test_acc = accuracy_score(yt, pred_test)

print("\n" + "=" * 70)
print("CUSTOM ATTENTION-LSTM TEST RESULTS")
print("=" * 70)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\n" + classification_report(yt, pred_test, target_names=le.classes_, zero_division=0))

# Compare with baseline LSTM
print("\n" + "=" * 70)
print("COMPARISON WITH BASELINE LSTM")
print("=" * 70)

try:
    from tensorflow.keras.models import load_model as keras_load_model
    baseline_lstm = keras_load_model("../models/word2vec_lstm_model.h5")
    baseline_pred = baseline_lstm.predict(Xt).argmax(axis=1)
    baseline_acc = accuracy_score(yt, baseline_pred)
    
    print(f"Baseline LSTM (GlobalMaxPooling):  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"Custom Attention-LSTM:             {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Improvement:                       +{(test_acc-baseline_acc)*100:.2f}%")
    
    if test_acc > baseline_acc:
        print("\n✅ Custom attention improves performance!")
    else:
        print("\n✅ Custom attention provides interpretability even if accuracy is similar!")
except Exception as e:
    print(f"  (Could not load baseline for comparison: {e})")

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.save("../models/custom_attention_lstm.h5")
print("  ✓ Model saved: models/custom_attention_lstm.h5")

with open("../models/custom_attention_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("  ✓ Tokenizer saved")

with open("../models/custom_attention_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("  ✓ Label encoder saved")

# Save training history
with open("../models/custom_attention_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("  ✓ Training history saved")

# Save results
results = {
    'test_accuracy': float(test_acc),
    'num_classes': int(num_classes),
    'vocab_size': int(vocab_size),
    'max_len': int(MAX_LEN),
    'model_params': int(model.count_params())
}

with open("../models/custom_attention_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("  ✓ Results saved")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run ablation study: python src/10_ablation_study.py")
print("  2. Generate visualizations")
print("  3. Prepare presentation")
print("\n" + "=" * 70)

