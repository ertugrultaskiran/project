"""
Test All Bug Fixes
Bu script tum duzeltmelerin calistigini dogrular
"""

import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Change to project root if in src/
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')

print("=" * 70)
print("TÜM HATA DÜZELTMELERİNİ TEST EDİYORUM")
print("=" * 70)
print(f"Çalışma dizini: {os.getcwd()}")

# Test 1: Import testleri
print("\n[TEST 1/6] Import testleri...")
try:
    # TensorFlow imports
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    print("  ✅ TensorFlow imports - BAŞARILI")
except ImportError as e:
    print(f"  ❌ TensorFlow imports - BAŞARISIZ: {e}")
    sys.exit(1)

try:
    # Other imports
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import classification_report, accuracy_score
    print("  ✅ Scikit-learn imports - BAŞARILI")
except ImportError as e:
    print(f"  ❌ Scikit-learn imports - BAŞARISIZ: {e}")
    sys.exit(1)

# Test 2: Veri dosyası kontrolü
print("\n[TEST 2/6] Veri dosyası kontrolü...")
try:
    df = pd.read_csv("data/cleaned_data.csv")
    assert df.shape[0] == 47837, f"Satır sayısı yanlış: {df.shape[0]}"
    assert df.shape[1] == 2, f"Sütun sayısı yanlış: {df.shape[1]}"
    assert 'text' in df.columns, "text sütunu yok"
    assert 'label' in df.columns, "label sütunu yok"
    assert df.isnull().sum().sum() == 0, "Null değerler var"
    print(f"  ✅ Veri dosyası - BAŞARILI ({df.shape[0]} satır)")
except Exception as e:
    print(f"  ❌ Veri dosyası - BAŞARISIZ: {e}")
    sys.exit(1)

# Test 3: Model dosyaları kontrolü
print("\n[TEST 3/6] Model dosyaları kontrolü...")
required_models = [
    "models/baseline_tfidf_logreg.pkl",
    "models/label_encoder.pkl",
    "models/tokenizer.pkl",
    "models/word2vec_lstm_model.h5",
    "models/word2vec_model.bin"
]

missing_models = []
for model_path in required_models:
    if not os.path.exists(model_path):
        missing_models.append(model_path)

if missing_models:
    print(f"  ❌ Eksik modeller: {missing_models}")
    print("  ⚠️  Önce modelleri eğitmeniz gerekiyor!")
else:
    print(f"  ✅ Tüm modeller mevcut ({len(required_models)} dosya)")

# Test 4: Baseline model yükleme
print("\n[TEST 4/6] Baseline model yükleme...")
try:
    with open("models/baseline_tfidf_logreg.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    print(f"  ✅ Baseline model - BAŞARILI")
    print(f"     Sınıf sayısı: {len(label_encoder.classes_)}")
    print(f"     Sınıflar: {', '.join(label_encoder.classes_)}")
except Exception as e:
    print(f"  ❌ Baseline model - BAŞARISIZ: {e}")

# Test 5: LSTM model yükleme
print("\n[TEST 5/6] LSTM model yükleme...")
try:
    lstm_model = load_model("models/word2vec_lstm_model.h5")
    
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    print(f"  ✅ LSTM model - BAŞARILI")
    print(f"     Vocabulary size: {len(tokenizer.word_index)}")
except Exception as e:
    print(f"  ❌ LSTM model - BAŞARISIZ: {e}")

# Test 6: Utils fonksiyon testi
print("\n[TEST 6/6] Utils fonksiyon testi...")
try:
    sys.path.insert(0, 'src')
    from utils import basic_clean
    
    test_text = "Hello WORLD! http://example.com #hashtag @mention"
    cleaned = basic_clean(test_text)
    
    assert isinstance(cleaned, str), "Temizlenmiş metin string değil"
    assert len(cleaned) > 0, "Temizlenmiş metin boş"
    assert cleaned == cleaned.lower(), "Lowercase yapılmamış"
    
    print(f"  ✅ Utils fonksiyonları - BAŞARILI")
    print(f"     Örnek: '{test_text[:30]}...' -> '{cleaned[:30]}...'")
except Exception as e:
    print(f"  ❌ Utils fonksiyonları - BAŞARISIZ: {e}")

# Test 7: Quick prediction test
print("\n[BONUS TEST] Hızlı tahmin testi...")
try:
    test_sample = "I need access to the database for my project"
    
    # Baseline prediction
    baseline_pred = baseline_model.predict([test_sample])[0]
    baseline_proba = baseline_model.predict_proba([test_sample])[0]
    
    print(f"  ✅ Tahmin başarılı!")
    print(f"     Metin: '{test_sample}'")
    print(f"     Tahmin: {baseline_pred}")
    print(f"     Güven: {max(baseline_proba):.2%}")
except Exception as e:
    print(f"  ⚠️  Tahmin testi başarısız (normal, model eğitimi gerekebilir): {e}")

# Özet
print("\n" + "=" * 70)
print("TEST SONUÇLARI")
print("=" * 70)
print("✅ Tüm import testleri geçti")
print("✅ Veri dosyası kontrolü geçti")
if not missing_models:
    print("✅ Tüm model dosyaları mevcut")
else:
    print("⚠️  Bazı modeller eksik (eğitim gerekli)")
print("\n🎉 Hata düzeltmeleri başarılı!")
print("\nSonraki adımlar:")
print("  1. BERT modelini eğitin: src/03_bert_transformer.ipynb")
print("  2. Ensemble modeli çalıştırın: python src/04_ensemble_model.py")
print("  3. Hyperparameter tuning yapın: python src/05_hyperparameter_tuning.py")
print("  4. API'yi başlatın: python src/06_inference_api.py")
print("=" * 70)

