"""
BERT Fine-Tuning Script
Doğrudan çalıştırılabilir Python scripti
Not: GPU kullanımı şiddetle önerilir!

Kullanım:
    python src/train_bert.py
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BERT FINE-TUNING FOR TICKET CLASSIFICATION")
print("=" * 70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Using device: {device}")

if device.type == 'cpu':
    print("\n⚠️  WARNING: GPU not detected! Training will be VERY slow.")
    print("   Consider using Google Colab or Kaggle for free GPU.")
    response = input("\n   Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("   Exiting...")
        exit()

# 1. Load Data
print("\n[1/9] Loading data...")
df = pd.read_csv("data/cleaned_data.csv")
print(f"   ✓ Data shape: {df.shape}")

# 2. Split Data
print("\n[2/9] Splitting data...")
X_train, X_tmp, y_train, y_tmp = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)
print(f"   ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 3. Label Encoding
print("\n[3/9] Encoding labels...")
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])
y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
num_labels = len(label_encoder.classes_)
print(f"   ✓ Number of classes: {num_labels}")

# 4. Dataset Class
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("\n[4/9] Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("   ✓ Tokenizer loaded!")

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16  # Reduce to 8 if GPU memory is low
EPOCHS = 3
LEARNING_RATE = 2e-5

print(f"\n   Hyperparameters:")
print(f"   - Max Length: {MAX_LENGTH}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Learning Rate: {LEARNING_RATE}")

# 5. Create DataLoaders
print("\n[5/9] Creating dataloaders...")
train_dataset = TicketDataset(X_train, y_train_encoded, tokenizer, MAX_LENGTH)
val_dataset = TicketDataset(X_val, y_val_encoded, tokenizer, MAX_LENGTH)
test_dataset = TicketDataset(X_test, y_test_encoded, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f"   ✓ Train batches: {len(train_loader)}")
print(f"   ✓ Val batches: {len(val_loader)}")
print(f"   ✓ Test batches: {len(test_loader)}")

# 6. Initialize Model
print("\n[6/9] Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
)
model = model.to(device)
print(f"   ✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# 7. Setup Optimizer and Scheduler
print("\n[7/9] Configuring optimizer...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
print(f"   ✓ Total training steps: {total_steps}")

# Training and Evaluation Functions
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item()
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader), predictions, true_labels

# 8. Train Model
print("\n[8/9] Starting training...")
print("=" * 70)

history = {
    'train_acc': [],
    'train_loss': [],
    'val_acc': [],
    'val_loss': []
}

best_val_acc = 0

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 70)
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)')
    
    val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)')
    
    history['train_acc'].append(train_acc.item())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.item())
    history['val_loss'].append(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/bert_model.pt')
        print(f'✓ Model saved! Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')

print("\n" + "=" * 70)
print("TRAINING COMPLETED!")
print("=" * 70)

# 9. Evaluate on Test Set
print("\n[9/9] Evaluating on test set...")
model.load_state_dict(torch.load('models/bert_model.pt'))

test_acc, test_loss, test_predictions, test_true_labels = eval_model(model, test_loader, device)

print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(
    test_true_labels, 
    test_predictions, 
    target_names=label_encoder.classes_,
    zero_division=0
))

# Save artifacts
print("\nSaving artifacts...")
with open('models/bert_training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
tokenizer.save_pretrained('models/bert_tokenizer')
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n✓ Saved files:")
print("  - models/bert_model.pt")
print("  - models/bert_training_history.pkl")
print("  - models/bert_tokenizer/")
print("  - models/label_encoder.pkl")

print("\n" + "=" * 70)
print("ALL DONE! 🎉")
print("=" * 70)

# Summary
print("\n📊 SUMMARY:")
print(f"  Best Model: BERT Fine-tuned")
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(f"  Training Time: {EPOCHS} epochs")
print(f"  Device: {device}")
print("\n✨ Model ready for production!")


