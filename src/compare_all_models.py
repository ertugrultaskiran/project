"""
Tüm Modelleri Karşılaştır: Baseline, LSTM, Ensemble, BERT
"""

import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('.')
from src.utils import basic_clean

print("=" * 70)
print("TÜM MODELLERİN KAPSAMLI KARŞILAŞTIRMASI")
print("=" * 70)

# Load data
df = pd.read_csv("data/cleaned_data.csv")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

print(f"\nTest set size: {len(X_test)}")

# Load models and artifacts
print("\nLoading models...")
with open("models/baseline_tfidf_logreg.pkl", "rb") as f:
    baseline_model = pickle.load(f)

lstm_model = load_model("models/word2vec_lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load BERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)
)
bert_model.load_state_dict(torch.load('models/bert_model.pt'))
bert_model = bert_model.to(device)
bert_model.eval()

print("✓ Tüm modeller yüklendi!")

# 1. BASELINE
print("\n[1/4] Baseline evaluating...")
baseline_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_probs = baseline_model.predict_proba(X_test)

# 2. LSTM
print("[2/4] LSTM evaluating...")
X_test_clean = X_test.apply(basic_clean)
sequences = tokenizer.texts_to_sequences(X_test_clean)
padded = pad_sequences(sequences, maxlen=80, padding="post", truncating="post")
lstm_probs = lstm_model.predict(padded, verbose=0)
lstm_pred_idx = lstm_probs.argmax(axis=1)
lstm_pred = label_encoder.inverse_transform(lstm_pred_idx)
lstm_acc = accuracy_score(y_test, lstm_pred)

# 3. ENSEMBLE
print("[3/4] Ensemble evaluating...")
ensemble_probs = (0.5 * baseline_probs + 0.5 * lstm_probs)
ensemble_pred_idx = np.argmax(ensemble_probs, axis=1)
ensemble_pred = label_encoder.inverse_transform(ensemble_pred_idx)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

# 4. BERT
print("[4/4] BERT evaluating...")
class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

bert_dataset = SimpleDataset(X_test, bert_tokenizer)
bert_loader = DataLoader(bert_dataset, batch_size=16)

bert_predictions = []
with torch.no_grad():
    for batch in bert_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        bert_predictions.extend(preds.cpu().numpy())

bert_pred = label_encoder.inverse_transform(bert_predictions)
bert_acc = accuracy_score(y_test, bert_pred)

# RESULTS
print("\n" + "=" * 70)
print("MODEL KARŞILAŞTIRMASI - FINAL RESULTS")
print("=" * 70)

results = pd.DataFrame({
    'Model': ['Baseline (TF-IDF+LogReg)', 'LSTM (Word2Vec+BiLSTM)', 'Ensemble (Base+LSTM)', 'BERT Fine-tuned'],
    'Accuracy': [baseline_acc, lstm_acc, ensemble_acc, bert_acc],
    'Accuracy %': [f'{baseline_acc*100:.2f}%', f'{lstm_acc*100:.2f}%', f'{ensemble_acc*100:.2f}%', f'{bert_acc*100:.2f}%']
}).sort_values('Accuracy', ascending=False)

print("\n" + results.to_string(index=False))

print("\n" + "=" * 70)
print("🏆 EN İYİ MODEL:", results.iloc[0]['Model'])
print(f"🎯 EN İYİ ACCURACY: {results.iloc[0]['Accuracy %']}")
print("=" * 70)

# Save results
results.to_csv('reports/final_model_comparison.csv', index=False)
print("\n✓ Sonuçlar kaydedildi: reports/final_model_comparison.csv")

# Detailed comparison
print("\n" + "=" * 70)
print("DETAYLI SINIF BAZINDA KARŞILAŞTIRMA")
print("=" * 70)

for model_name, preds in [('Baseline', baseline_pred), ('LSTM', lstm_pred), 
                           ('Ensemble', ensemble_pred), ('BERT', bert_pred)]:
    print(f"\n{model_name}:")
    print(classification_report(y_test, preds, zero_division=0, digits=3))

print("\n" + "=" * 70)
print("KARŞILAŞTIRMA TAMAMLANDI!")
print("=" * 70)

