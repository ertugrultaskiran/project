# 🚀 BERT Model Eğitim Rehberi

## 📋 Seçenekler

BERT modelini 3 farklı yolla eğitebilirsiniz:

---

## 1️⃣ Jupyter Notebook (Önerilen - Adım Adım)

### Avantajlar:
- ✅ Her adımı görebilirsiniz
- ✅ Ara sonuçları inceleyebilirsiniz
- ✅ İstediğiniz yerde durup devam edebilirsiniz

### Kullanım:
```bash
jupyter notebook src/03_bert_transformer.ipynb
```

Ardından browser'da notebook'u açın ve hücreleri sırayla çalıştırın.

---

## 2️⃣ Python Script (Kolay - Otomatik)

### Avantajlar:
- ✅ Tek komutla başlatıp bitirirsiniz
- ✅ Terminal'de ilerlemeyi görebilirsiniz
- ✅ Daha hızlı ve pratik

### Kullanım:
```bash
python src/train_bert.py
```

Script otomatik olarak:
1. Veriyi yükler
2. BERT modelini indirir
3. Eğitimi başlatır
4. En iyi modeli kaydeder
5. Test sonuçlarını gösterir

---

## 3️⃣ Google Colab (GPU Yoksa - ÜCRETSİZ GPU!)

### Avantajlar:
- ✅ Ücretsiz GPU kullanımı
- ✅ Kendi bilgisayarınız yormaz
- ✅ Çok daha hızlı (~10-20x)

### Adımlar:

1. **Google Colab'a Git:**
   - https://colab.research.google.com

2. **Notebook'u Yükle:**
   - File → Upload notebook
   - `src/03_bert_transformer.ipynb` dosyasını seçin

3. **GPU'yu Aktifleştir:**
   - Runtime → Change runtime type
   - Hardware accelerator → **GPU**
   - Save

4. **Veriyi Yükle:**
   ```python
   # İlk hücreye ekle:
   from google.colab import files
   uploaded = files.upload()  # cleaned_data.csv'yi yükle
   ```

5. **Çalıştır:**
   - Runtime → Run all

---

## ⚙️ Eğitim Parametreleri

### Varsayılan Ayarlar:
```python
MAX_LENGTH = 128        # Metin uzunluğu
BATCH_SIZE = 16         # Her adımda işlenen örnek sayısı
EPOCHS = 3              # Eğitim döngüsü sayısı
LEARNING_RATE = 2e-5    # Öğrenme hızı
```

### GPU Memory Düşükse:
```python
BATCH_SIZE = 8   # veya 4
MAX_LENGTH = 64  # daha kısa metinler
```

### Daha İyi Sonuç İçin:
```python
EPOCHS = 5       # Daha fazla eğitim
BATCH_SIZE = 32  # Daha büyük batch (GPU yeterliyse)
```

---

## 📊 Beklenen Sonuçlar

### Eğitim Süresi:
- **CPU:** ~8-12 saat ⚠️ (ÇOK YAVAŞ!)
- **GPU (Google Colab):** ~2-3 saat ✅
- **High-end GPU:** ~30-60 dakika ⚡

### Beklenen Accuracy:
- **Epoch 1:** ~85-88%
- **Epoch 2:** ~88-91%
- **Epoch 3:** ~90-93% 🎯

### Memory Kullanımı:
- **CPU:** ~4-8 GB RAM
- **GPU:** ~4-6 GB VRAM
- **Colab Free:** Yeterli (12 GB VRAM)

---

## 🎯 Eğitim Sırasında

### İlerlemeyi Takip Edin:
```
Epoch 1/3
----------------------------------------------------------------------
Training: 100%|██████████| 2392/2392 [45:23<00:00, 0.88it/s, loss=0.2341]
Train Loss: 0.2341 | Train Acc: 0.9123 (91.23%)
Evaluating: 100%|██████████| 299/299 [03:12<00:00, 1.56it/s]
Val Loss: 0.1823 | Val Acc: 0.9287 (92.87%)
✓ Model saved! Best Val Acc: 0.9287 (92.87%)
```

### Sorun Yaşarsanız:

#### "CUDA out of memory"
```python
# Batch size'ı küçült:
BATCH_SIZE = 8  # veya 4
```

#### "Too slow on CPU"
- Google Colab kullanın (ücretsiz GPU)
- Veya cloud GPU servisleri (AWS, Azure)

#### "Model not improving"
- Learning rate'i ayarlayın: `LEARNING_RATE = 1e-5`
- Daha fazla epoch: `EPOCHS = 5`

---

## 💾 Oluşturulacak Dosyalar

Eğitim sonunda:
```
models/
├── bert_model.pt                  ← Ana model (400+ MB)
├── bert_training_history.pkl      ← Eğitim grafiği
└── bert_tokenizer/                ← BERT tokenizer
    ├── config.json
    ├── tokenizer_config.json
    └── vocab.txt
```

---

## 🧪 Eğitim Sonrası Test

### 1. Model Performansını Kontrol Et:
```bash
python src/evaluate_current_models.py
```

Şimdi BERT de dahil olacak!

### 2. Tüm Modelleri Karşılaştır:
```bash
jupyter notebook src/08_quick_evaluation.ipynb
```

### 3. Production API'de Kullan:
API'yi güncelleyip BERT endpoint'i ekleyebilirsiniz.

---

## 📈 Model Karşılaştırması (Beklenen)

```
Model              | Accuracy | İyileştirme
-------------------|----------|-------------
Baseline           | 86.04%   | Referans
LSTM               | 87.00%   | +0.96%
Ensemble           | 88.40%   | +2.36%
BERT (Beklenen)    | 90-93%   | +4-7% 🎯
```

---

## ⚠️ Önemli Notlar

1. **İlk çalıştırma:**
   - BERT modelini internet'ten indirecek (~400 MB)
   - İlk epoch daha yavaş olabilir (cache oluşturuyor)

2. **Checkpoint:**
   - Her epoch sonunda en iyi model kaydedilir
   - Eğitim kesilirse en son kaydedilen model kullanılır

3. **Overfitting:**
   - Val accuracy düşmeye başlarsa durdurun
   - 3 epoch genelde yeterli

4. **Results:**
   - Test accuracy > 90% hedefimiz
   - Her sınıf için F1-score > 0.85 ideal

---

## 🚀 Hızlı Başlangıç

### En Basit Yol (Python Script):
```bash
# 1. Script'i çalıştır
python src/train_bert.py

# 2. Bekle (~2-3 saat GPU ile)

# 3. Sonuçları gör
# Test Accuracy: 91.23% (örnek)

# 4. Modeli kullan
python src/06_inference_api.py
```

### Google Colab ile (Önerilen - GPU Yoksa):
1. https://colab.research.google.com adresine git
2. `03_bert_transformer.ipynb` dosyasını yükle
3. Runtime → Change runtime → GPU
4. Runtime → Run all
5. Bekle (~2-3 saat)
6. Model'i indir ve `models/` klasörüne koy

---

## ✅ Başarı Kriterleri

Eğitim başarılı sayılır:
- ✅ Test Accuracy > 90%
- ✅ Val Accuracy ile Test Accuracy arası fark < 2%
- ✅ Tüm sınıflar için F1-score > 0.80
- ✅ Model dosyası oluşturuldu

---

## 🆘 Yardım

Sorun yaşarsanız:
1. GPU kullandığınızdan emin olun
2. Batch size'ı küçültün
3. Google Colab deneyin
4. `BERT_EGITIM_SORUNLARI.md` dosyasına bakın

---

**Hazırsınız! Şimdi modelinizi eğitin! 🚀**

**Tahmini Süre:** 2-3 saat (GPU ile)  
**Beklenen Sonuç:** %90-93 accuracy  
**Zorluk:** Orta (GPU gerekli)


