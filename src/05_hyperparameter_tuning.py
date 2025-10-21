"""
Hyperparameter Tuning for LSTM Model
RandomSearch ile en iyi parametreleri bul
NOT: Bu script, LSTM modeliniz için en iyi hiperparametreleri bulmak amacıyla kullanılır.
Önceden eğitilmiş bir Word2Vec modeli ve hazırlanmış veri gerektirir.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import sys
sys.path.append('..')
from utils import basic_clean

print("=" * 60)
print("HYPERPARAMETER TUNING - LSTM MODEL")
print("=" * 60)

# 1. LOAD DATA
print("\n[1/6] Veri yükleniyor...")
df = pd.read_csv("../data/cleaned_data.csv")
df['text_clean'] = df['text'].apply(basic_clean)
print(f"[OK] Veri yüklendi: {df.shape}")

# 2. TRAIN/VAL/TEST SPLIT
print("\n[2/6] Veri bölünüyor...")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)
print(f"[OK] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 3. LOAD PRETRAINED MODELS
print("\n[3/6] Tokenizer ve Word2Vec yükleniyor...")
with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
with open("../models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

w2v = Word2Vec.load("../models/word2vec_model.bin")
print("[OK] Modeller yüklendi")

# 4. PREPARE SEQUENCES
print("\n[4/6] Sequences hazırlanıyor...")
MAX_LEN = 80
seq_train = tokenizer.texts_to_sequences(X_train)
seq_val = tokenizer.texts_to_sequences(X_val)

Xtr = pad_sequences(seq_train, maxlen=MAX_LEN, padding="post", truncating="post")
Xv = pad_sequences(seq_val, maxlen=MAX_LEN, padding="post", truncating="post")

ytr = label_encoder.transform(y_train)
yv = label_encoder.transform(y_val)

print(f"[OK] Train shape: {Xtr.shape}")

# 5. CREATE EMBEDDING MATRIX
print("\n[5/6] Embedding matrix oluşturuluyor...")
vocab_size = min(len(tokenizer.word_index) + 1, 40000)
emb_dim = 200
num_classes = len(label_encoder.classes_)

embedding_matrix = np.zeros((vocab_size, emb_dim))
for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        break
    if word in w2v.wv:
        embedding_matrix[i] = w2v.wv[word]

print(f"[OK] Embedding matrix: {embedding_matrix.shape}")

# 6. HYPERPARAMETER SEARCH
print("\n[6/6] Hyperparameter tuning başlıyor...")

# Test different configurations manually
param_configs = [
    {'lstm_units': 128, 'dropout_rate': 0.3, 'lr': 0.001, 'name': 'Baseline'},
    {'lstm_units': 256, 'dropout_rate': 0.3, 'lr': 0.001, 'name': 'More Units'},
    {'lstm_units': 128, 'dropout_rate': 0.4, 'lr': 0.001, 'name': 'More Dropout'},
    {'lstm_units': 128, 'dropout_rate': 0.2, 'lr': 0.001, 'name': 'Less Dropout'},
    {'lstm_units': 128, 'dropout_rate': 0.3, 'lr': 0.0005, 'name': 'Lower LR'},
    {'lstm_units': 64, 'dropout_rate': 0.3, 'lr': 0.001, 'name': 'Smaller Model'},
]

best_val_acc = 0
best_config = None
results = []

for i, config in enumerate(param_configs, 1):
    print(f"\n--- Testing Config {i}/{len(param_configs)}: {config['name']} ---")
    print(f"    LSTM Units: {config['lstm_units']}, Dropout: {config['dropout_rate']}, LR: {config['lr']}")
    
    # Build model
    inp = layers.Input(shape=(MAX_LEN,))
    emb = layers.Embedding(vocab_size, emb_dim, weights=[embedding_matrix], trainable=False)(inp)
    x = layers.SpatialDropout1D(config['dropout_rate'])(emb)
    x = layers.Bidirectional(layers.LSTM(config['lstm_units'], return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inp, out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Train
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        epochs=10,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate
    val_acc = max(history.history['val_accuracy'])
    print(f"    Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    results.append({
        'name': config['name'],
        'lstm_units': config['lstm_units'],
        'dropout_rate': config['dropout_rate'],
        'learning_rate': config['lr'],
        'val_accuracy': val_acc
    })
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_config = config
        print(f"    *** NEW BEST! ***")

# 7. RESULTS
print(f"\n{'='*60}")
print("HYPERPARAMETER TUNING SONUÇLARI")
print(f"{'='*60}")

results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False)
print("\nTüm Konfigürasyonlar:")
print(results_df.to_string(index=False))

print(f"\n{'='*60}")
print("EN İYİ KONFİGÜRASYON")
print(f"{'='*60}")
print(f"Name: {best_config['name']}")
print(f"LSTM Units: {best_config['lstm_units']}")
print(f"Dropout Rate: {best_config['dropout_rate']}")
print(f"Learning Rate: {best_config['lr']}")
print(f"Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# Save results
results_df.to_csv('../reports/hyperparameter_tuning_results.csv', index=False)
print(f"\n[OK] Results saved: reports/hyperparameter_tuning_results.csv")

with open("../models/best_hyperparameters.pkl", "wb") as f:
    pickle.dump(best_config, f)
print(f"[OK] Best config saved: models/best_hyperparameters.pkl")

print(f"\n{'='*60}")
print("HYPERPARAMETER TUNING TAMAMLANDI!")
print(f"{'='*60}")

