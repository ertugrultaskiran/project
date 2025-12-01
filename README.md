# 🎯 Intelligent Ticket Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Bitirme Projesi** · 47 837 gerçek IT destek talebini 8 kategoriye ayıran, uçtan uca NLP + Deep Learning çözümü

## 📋 Proje Özeti

Bu sistem, şirket içi destek taleplerini otomatik sınıflandırmak için klasik ML, derin öğrenme ve transfer öğrenmeyi birleştirir. Veri temizleme, model eğitimi, değerlendirme, ablation study, REST API ve web demosu tek bir repo altında toplanmıştır.

- **Veri**: 47 837 satır, 8 sınıf (ticket kategorisi)
- **Diller/Kütüphaneler**: Python 3.8+, TensorFlow/Keras, PyTorch, Transformers, scikit-learn, Flask
- **Donanım**: GPU opsiyonel (BERT ve LSTM için önerilir)

### 🏆 Başarılar
- ✅ 47 837 ticket temizlendi ve etiketlendi
- ✅ 8 kategori için dengeli veri pipeline’ı
- ✅ %86.04 test accuracy — TF-IDF + Logistic Regression
- ✅ %87.00 test accuracy — Word2Vec + BiLSTM
- ✅ %88.40 test accuracy — Baseline + LSTM Ensemble
- ✅ %88.82 test accuracy — BERT fine-tuning (PyTorch)
- ✅ Production-ready REST API + web demo
- ✅ Docker, otomasyon scriptleri ve ayrıntılı akademik dokümantasyon

### 🚀 Kullanılan Yaklaşımlar
1. **Baseline (TF-IDF + Logistic Regression)** – sağlam referans metriği
2. **Deep Learning (Word2Vec + Bidirectional LSTM)** – sıralı bağlamı yakalar
3. **Transfer Learning (BERT Fine-tuning)** – yüksek doğruluklu transformer tabanı
4. **Ensemble (Baseline + LSTM)** – sınıf bazlı hata azalması
5. **Custom Attention + Domain Features** – akademik katkı ve açıklanabilirlik

## 📂 Proje Klasör Yapısı

```
project/
├── data/                # Ham ve temizlenmiş veri
├── models/              # Eğitilmiş modeller + tokenizer/encoder
├── reports/             # PNG & CSV çıktıları (karşılaştırmalar, loglar)
├── src/                 # Notebooklar, eğitim scriptleri, API, web app
│   ├── static/, templates/  # Web uygulaması varlıkları
│   ├── 04_ensemble_model.py
│   ├── 06_inference_api.py
│   ├── 07_model_evaluation.py
│   ├── 10_ablation_study.py
│   └── utils.py
├── docs/                # Tüm rehberler, akademik rapor, sunum notları
├── scripts/             # Bash otomasyon komutları
├── tests/               # Pytest senaryoları
├── README.md            # Bu dosya
├── FINAL_SUMMARY.md     # Hızlı teslimat özeti
├── QUICK_START_1-2_DAYS.md
├── requirements.txt · config.yaml · Dockerfile
└── START_WEB_APP.bat · RUN_WEB_APP.py · test_flask.py
```

> Detaylı dokümantasyon ve kurulum rehberleri `docs/` klasöründe gruplanmıştır.

## 🚀 Kurulum

### 1. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Jupyter Notebook'u Başlatın

```bash
cd src
jupyter notebook
```

## 📊 Çalıştırma Adımları

### Adım 1: Veri Kontrolü ve Hazırlık
**Dosya**: `src/00_check_data.ipynb`

- Veriyi yükler ve sütun isimlerini kontrol eder
- Null/boş değerleri temizler
- Sınıf dağılımını analiz eder
- Temizlenmiş veriyi `data/cleaned_data.csv` olarak kaydeder

**Kontrol Listesi:**
- ✓ Sütun isimleri doğru mu?
- ✓ Null / boş ve çok kısa metinler temizlendi mi?
- ✓ Sınıf sayısı makul mü? (5-10 sınıf ideal)

### Adım 2: Baseline Model (TF-IDF + Logistic Regression)
**Dosya**: `src/01_baseline_tfidf_logreg.ipynb`

- Train/validation/test bölümü yapar (stratified split)
- TF-IDF vektörizasyonu uygular
- Logistic Regression modeli eğitir
- Class weight ile dengesiz sınıfları ele alır
- Model performansını değerlendirir

**Bu baseline, "derine" inmeden önce sağlam bir referans metrik sağlar.**

### Adım 3: Word2Vec + LSTM Derin Öğrenme Modeli
**Dosya**: `src/02_word2vec_lstm.ipynb`

**Alt Adımlar:**

#### 3.1 Metin Temizleme
- `utils.py`'daki `basic_clean()` fonksiyonu ile temel temizlik
- URL'ler, özel karakterler temizlenir
- İngilizce için minimal temizlik (aşırı temizlik anlam kaybettirir)

#### 3.2 Tokenization
- Keras Tokenizer ile kelime indeksi oluşturur
- MAX_VOCAB = 40,000 kelime
- Metinleri sayı dizilerine çevirir
- Padding ile sabit uzunlukta vektörler (MAX_LEN=80)

#### 3.3 Word2Vec Eğitimi
- Gensim Word2Vec ile kelime embedding'leri öğrenir
- vector_size=200, window=5, sg=1 (skip-gram)
- Sadece train seti üzerinde eğitilir

#### 3.4 Embedding Matrisi
- Word2Vec'ten öğrenilen vektörler embedding matrisine dönüştürülür
- Bilinmeyen kelimeler için random initialization

#### 3.5 LSTM Model Mimarisi
```
Input (MAX_LEN)
  ↓
Embedding Layer (trainable=False, Word2Vec weights)
  ↓
SpatialDropout1D (0.2)
  ↓
Bidirectional LSTM (128 units, return_sequences=True)
  ↓
GlobalMaxPooling1D
  ↓
Dropout (0.3)
  ↓
Dense (num_classes, softmax)
```

#### 3.6 Eğitim
- Class weights ile dengesiz sınıf problemi çözülür
- EarlyStopping (patience=3, monitor=val_accuracy)
- epochs=15, batch_size=64
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy

#### 3.7 Değerlendirme
- Validation ve Test setleri üzerinde performans ölçülür
- Classification report ile detaylı metrikler
- Accuracy, Precision, Recall, F1-Score

#### 3.8 Model Kaydetme
- LSTM modeli: `models/word2vec_lstm_model.h5`
- Word2Vec: `models/word2vec_model.bin`
- Tokenizer: `models/tokenizer.pkl`
- Label Encoder: `models/label_encoder.pkl`

## 📈 Sonuçlar

`reports/final_model_comparison.csv` dosyasından özetlenen en güncel test sonuçları:

| Model | Test Accuracy |
|-------|---------------|
| Baseline (TF-IDF + LogReg) | **86.04 %** |
| Word2Vec + BiLSTM | **87.00 %** |
| Ensemble (Baseline + LSTM) | **88.40 %** |
| BERT Fine-tuned | **88.82 %** |

Ek çıktılar:
- `reports/model_comparison_confusion_matrix.png` – Baseline vs. LSTM hata dağılımı
- `reports/per_class_comparison.csv` – sınıf bazında precision/recall/F1
- `reports/training_history.png` – BiLSTM eğitim eğrileri
- `reports/02_improvement_progress.png` – ablation sonuçları

## 🔥 Son Dönem İyileştirmeleri (Kasım 2025)

1. **BERT Fine-tuning** – `src/03_bert_transformer.ipynb`
   - PyTorch + Transformers
   - 88.82 % test accuracy (raporlara işlendi)
2. **Ensemble Model** – `src/04_ensemble_model.py`
   - Baseline + LSTM ağırlık taraması, konfig kaydı `models/ensemble_config.pkl`
3. **Hyperparameter Tuning** – `src/05_hyperparameter_tuning.py`
   - Grid/Random search altyapısı, otomatik loglama
4. **Production REST API** – `src/06_inference_api.py`
   - `/health`, `/predict/baseline`, `/predict/lstm`, `/predict/ensemble`
   - Hazır model/tokenez yükleme ve top-3 tahmin dönüşü
5. **Model Evaluation Suite** – `src/07_model_evaluation.py` ve `src/evaluate_current_models.py`
   - Confusion matrix, ROC/PR eğrileri, hata analizi CSV’leri
6. **Dokümantasyon + Web Demo**
   - `docs/ACADEMIC_CONTRIBUTIONS.md`, `FINAL_SUMMARY.md`, `WEB_APP_README.md`
   - `src/web_app.py` + `START_WEB_APP.bat` ile canlı demo

### 📊 Çalıştırma Rehberi:

#### REST API Başlatma:
```bash
cd src
python 06_inference_api.py
```

API Endpoints:
- `GET /health` - Sağlık kontrolü
- `POST /predict/baseline` - Baseline model tahmin
- `POST /predict/lstm` - LSTM model tahmin
- `POST /predict/ensemble` - Ensemble model tahmin

#### Model Değerlendirme:
```bash
cd src
python evaluate_current_models.py
```

#### Ensemble Model Eğitimi:
```bash
cd src
python 04_ensemble_model.py
```

#### Hyperparameter Tuning:
```bash
cd src
python 05_hyperparameter_tuning.py
```

## 🌟 Original Contributions

1. **Custom Attention Mechanism**
   - Dosya: `src/custom_attention_layer.py`
   - Tamamen sıfırdan yazıldı; TensorFlow/Keras tabanlı
   - Açıklanabilirlik için ağırlık vektörleri dışa aktarılabilir
2. **Domain-Specific Feature Engineering**
   - Dosya: `src/custom_features.py`
   - 20+ IT ticket özelliği (access, hardware, network vb.)
   - `ITTicketFeatureExtractor` ile batch üretim
3. **Comprehensive Ablation Study**
   - Dosya: `src/10_ablation_study.py`
   - TF-IDF, LSTM, domain feature ve ensemble katkıları nicel ölçümlendi
4. **Hybrid / Ensemble Yaklaşımı**
   - `src/04_ensemble_model.py` + custom attention modeli (`src/11_train_custom_attention.py`)
   - Klasik + derin öğrenme + domain feature birleşimi

## 🚀 Quick Start for Academic Review (1-2 days)

See `QUICK_START_1-2_DAYS.md` for rapid deployment guide!

### Train Custom Attention Model:
```bash
python src/11_train_custom_attention.py
```

### Run Ablation Study:
```bash
python src/10_ablation_study.py
```

### Generate Visualizations:
```bash
python src/12_generate_visuals.py
```

### 🌐 **NEW: Launch Web Application Demo!**
```bash
START_WEB_APP.bat
```
Then open: `http://localhost:5000`

**Features:**
- 📊 Interactive dashboard
- 💬 Chatbot widget (bottom-right)
- 🎯 Real-time ticket classification
- 📋 Classification history panel
- 📈 Live statistics

Perfect for **live demonstrations** to professors!

## 📚 Academic Documentation

- **Contributions**: `docs/ACADEMIC_CONTRIBUTIONS.md`
- **Presentation Guide**: `docs/PRESENTATION_GUIDE.md`
- **Quick Start**: `QUICK_START_1-2_DAYS.md`

## 🎯 Gelecek İyileştirmeler

1. **Cross-Validation** – stratified K-fold ile daha sağlam metrikler
2. **Data Augmentation** – az temsil edilen sınıflar için paraphrase/back-translation
3. **Model Monitoring** – REST API loglarını MLflow veya Evidently ile takip
4. **A/B Testing** – LSTM vs. BERT canlı karşılaştırma
5. **Model Versioning** – MLflow veya DVC ile model/deney izleme

## 📝 Notlar

- **Baseline önce çalıştırılmalı**: TF-IDF + LogReg hızlı bir referans sağlar
- **LSTM uzun sürer**: GPU kullanımı önerilir (Google Colab, Kaggle)
- **Dengesiz sınıflar**: Class weights kullanıldı
- **Embedding trainable=False**: Word2Vec ağırlıkları sabit tutuldu (fine-tuning için True yapılabilir)

## 👨‍💻 Yazar

Bitirme Projesi - Topic Modelling with NLP

## 📚 Kaynaklar

- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## 📄 Lisans

Bu proje eğitim amaçlıdır.



