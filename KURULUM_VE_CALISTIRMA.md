# 🚀 Kurulum ve Çalıştırma Rehberi

## ✅ Sistem Durumu (21 Ekim 2025, 12:49)

**TÜM SİSTEMLER HAZIR! ✓**

Kurulu Kütüphaneler:
- ✅ TensorFlow: 2.20.0
- ✅ PyTorch: 2.9.0+cpu
- ✅ Transformers: 4.57.1
- ✅ Tüm bağımlılıklar

---

## 📦 Kurulum (İlk Kez)

### 1. Temel Kütüphaneleri Kur
```bash
pip install -r requirements.txt
```

### 2. BERT için Ek Kütüphaneler (Zaten kurulu ✓)
```bash
pip install torch transformers tqdm
```

---

## 🎯 Modelleri Çalıştırma

### 1. Mevcut Modelleri Test Et
```bash
cd src
python evaluate_current_models.py
```

**Beklenen Çıktı:**
- Baseline Model: ~86% accuracy
- LSTM Model: ~87% accuracy
- Detaylı classification report
- Confusion matrix grafikleri

---

### 2. BERT Modelini Eğit (Öncelik 1)

#### Jupyter Notebook ile:
```bash
jupyter notebook src/03_bert_transformer.ipynb
```

#### Doğrudan Python ile:
```bash
cd src
jupyter nbconvert --to script 03_bert_transformer.ipynb
python 03_bert_transformer.py
```

**Önemli Notlar:**
- ⚠️ GPU kullanımı şiddetle önerilir (CPU'da çok yavaş)
- Eğitim süresi: ~2-3 saat (GPU ile)
- Batch size: 16 (GPU memory'e göre ayarlayın)
- Epochs: 3
- Beklenen accuracy: **90-93%**

**GPU Yoksa:**
- Google Colab kullanın (ücretsiz GPU)
- Kaggle Notebooks kullanın
- Ya da daha küçük bir model deneyin

---

### 3. Ensemble Modeli Çalıştır (Öncelik 2)

```bash
cd src
python 04_ensemble_model.py
```

**Ne Yapar:**
- Baseline + LSTM modellerini birleştirir
- Farklı weight kombinasyonlarını test eder
- En iyi kombinasyonu otomatik seçer

**Beklenen Çıktı:**
- Test accuracy: ~89-90%
- Farklı weight kombinasyonlarının sonuçları
- En iyi model konfigürasyonu

---

### 4. Hyperparameter Tuning (Öncelik 3)

```bash
cd src
python 05_hyperparameter_tuning.py
```

**Ne Yapar:**
- LSTM modelini optimize eder
- Dropout, learning rate, LSTM units test eder
- En iyi parametreleri bulur

**Beklenen İyileştirme:** +1-2%

**Not:** Bu işlem uzun sürebilir (~1-2 saat)

---

### 5. REST API'yi Başlat

```bash
cd src
python 06_inference_api.py
```

**Endpoints:**
- `GET /health` - Sağlık kontrolü
- `POST /predict/baseline` - Baseline model tahmin
- `POST /predict/lstm` - LSTM model tahmin
- `POST /predict/ensemble` - Ensemble model tahmin

**Test:**
```bash
curl -X POST http://localhost:5000/predict/lstm \
  -H "Content-Type: application/json" \
  -d '{"text": "I need access to the database"}'
```

---

## 📊 Jupyter Notebook'ları Çalıştırma

### Tüm Notebook'lar:
1. `00_check_data.ipynb` - Veri kontrolü ve EDA ✅
2. `01_baseline_tfidf_logreg.ipynb` - Baseline model ✅
3. `02_word2vec_lstm.ipynb` - LSTM model ✅
4. `03_bert_transformer.ipynb` - BERT fine-tuning ✅ (Artık çalışır)
5. `08_quick_evaluation.ipynb` - Hızlı değerlendirme ✅

### Çalıştırma:
```bash
jupyter notebook
```

Ardından browser'da istediğiniz notebook'u açın ve çalıştırın.

---

## 🐛 Sorun Giderme

### "Import torch could not be resolved"
**Çözüm:** ✅ Zaten çözüldü!
```bash
pip install torch transformers
```

### "Import tensorflow.keras.models could not be resolved"
**Çözüm:** ✅ Zaten çözüldü!
- Bu sadece linter uyarısıdır
- Kod çalışır durumda
- `.vscode/settings.json` ile uyarılar kapatıldı

### GPU Bulunamıyor (BERT için)
**Çözüm:**
```python
# CPU kullanımını zorunlu kıl
device = torch.device('cpu')
```
Ya da Google Colab/Kaggle kullanın

### Memory Hatası (BERT eğitimi)
**Çözüm:**
- Batch size'ı küçült: `BATCH_SIZE = 8` veya `BATCH_SIZE = 4`
- Max length'i küçült: `MAX_LENGTH = 64`

---

## 📁 Proje Yapısı

```
project/
├── data/
│   ├── cleaned_data.csv              ✅ Hazır
│   └── all_tickets_processed_improved_v3.xlsx
├── models/                            ✅ Modeller mevcut
│   ├── baseline_tfidf_logreg.pkl     ✅
│   ├── word2vec_lstm_model.h5        ✅
│   ├── word2vec_model.bin            ✅
│   ├── tokenizer.pkl                 ✅
│   └── label_encoder.pkl             ✅
├── src/
│   ├── 00_check_data.ipynb           ✅
│   ├── 01_baseline_tfidf_logreg.ipynb ✅
│   ├── 02_word2vec_lstm.ipynb        ✅
│   ├── 03_bert_transformer.ipynb     ✅ (Yeni eklendi)
│   ├── 04_ensemble_model.py          ✅ (Düzeltildi)
│   ├── 05_hyperparameter_tuning.py   ✅ (Düzeltildi)
│   ├── 06_inference_api.py           ✅
│   ├── 07_model_evaluation.py        ✅
│   ├── 08_quick_evaluation.ipynb     ✅
│   ├── evaluate_current_models.py    ✅
│   └── utils.py                      ✅
└── reports/                           ✅ Grafikler kaydedilecek
```

---

## 🎯 Önerilen Çalıştırma Sırası

### Kısa Yol (Hızlı Test):
```bash
# 1. Mevcut modelleri değerlendir
python src/evaluate_current_models.py

# 2. Ensemble modeli çalıştır
python src/04_ensemble_model.py

# 3. API'yi başlat
python src/06_inference_api.py
```

### Tam Yol (En İyi Sonuç):
```bash
# 1. Veriyi kontrol et
jupyter notebook src/00_check_data.ipynb

# 2. Baseline'ı kontrol et (zaten eğitilmiş)
jupyter notebook src/01_baseline_tfidf_logreg.ipynb

# 3. LSTM'i kontrol et (zaten eğitilmiş)
jupyter notebook src/02_word2vec_lstm.ipynb

# 4. BERT'i eğit (en önemli!)
jupyter notebook src/03_bert_transformer.ipynb

# 5. Ensemble oluştur
python src/04_ensemble_model.py

# 6. Hyperparameter tuning
python src/05_hyperparameter_tuning.py

# 7. Final değerlendirme
python src/evaluate_current_models.py

# 8. API'yi başlat
python src/06_inference_api.py
```

---

## 📈 Beklenen Sonuçlar

| Adım | Model | Accuracy | Durum |
|------|-------|----------|-------|
| 1 | Baseline (TF-IDF + LogReg) | ~86% | ✅ Eğitilmiş |
| 2 | LSTM (Word2Vec + BiLSTM) | ~87% | ✅ Eğitilmiş |
| 3 | BERT Fine-tuned | **90-93%** | ⏳ Eğitilecek |
| 4 | Ensemble (Baseline+LSTM) | **89-90%** | ⏳ Çalıştırılacak |
| 5 | Tuned LSTM | **88-89%** | ⏳ Optimize edilecek |

---

## ✅ Son Kontrol Listesi

Projeyi çalıştırmadan önce:

- [x] Python 3.8+ kurulu
- [x] requirements.txt'ten paketler kurulu
- [x] PyTorch ve Transformers kurulu
- [x] TensorFlow çalışıyor
- [x] Veri dosyaları mevcut (`data/cleaned_data.csv`)
- [x] Model dosyaları mevcut (`models/` dizini)
- [x] Linter ayarları yapıldı
- [x] Tüm import'lar çalışıyor

**Sonuç: HAZIR! 🚀**

---

## 🆘 Destek

Sorun yaşarsanız:
1. `BUGFIX_REPORT.md` dosyasını kontrol edin
2. `SORUN_COZUMLERI.md` dosyasını okuyun
3. `TAMAMLANDI.md` dosyasında detayları görün

---

**Son Güncelleme:** 21 Ekim 2025, 12:49  
**Durum:** ✅ TÜM SİSTEMLER HAZIR  
**Test:** Tüm kütüphaneler test edildi ve çalışıyor


