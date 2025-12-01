# 🇹🇷 Türkçe Destek Ekleme - Adım Adım Rehber

## ✅ Yapılan Değişiklikler (Otomatik Tamamlandı)

1. **80 Türkçe örnek ticket eklendi** → `data/turkish_tickets_professional.csv`
2. **BERT modeli çok dilli yapıldı** → `bert-base-multilingual-cased`
3. **Tokenizer güncellendi** → Türkçe karakterleri destekliyor
4. **Veri birleştirme scripti hazır** → `src/prepare_multilingual_data.py`

---

## 📋 ŞİMDİ YAPILACAKLAR (2-3 saat)

### Adım 1: Çok Dilli Veri Setini Oluştur (2 dakika)

```bash
cd src
python prepare_multilingual_data.py
```

**Beklenen çıktı:**
```
ÇOK DİLLİ VERİ SETİ HAZIRLAMA
İngilizce örnekler: 47,837
Türkçe örnekler: 80
Toplam örnekler: 47,917
✅ Veri hazırlama tamamlandı!
```

---

### Adım 2: BERT Modelini Eğit (1.5-2 saat)

```bash
cd src
jupyter notebook
```

**Jupyter'da:**
1. `03_bert_transformer.ipynb` dosyasını aç
2. **Tüm cell'leri çalıştır** (Kernel → Restart & Run All)
3. GPU ile ~1.5-2 saat sürecek
4. Model `models/bert_model.pt` olarak kaydedilecek

**Önemli:** Notebook zaten güncellendi, değişiklik yapman gerekmiyor!

---

### Adım 3: Web Uygulamasına BERT Ekle (15 dakika)

`src/web_app.py` dosyasını aç ve şu değişiklikleri yap:

#### Değişiklik 1: BERT modelini yükle (satır ~42 civarı)

**Ekle:**
```python
# BERT model (Multilingual)
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    
    bert_tokenizer = BertTokenizer.from_pretrained("../models/bert_tokenizer")
    bert_model = BertForSequenceClassification.from_pretrained("../models/bert_model.pt")
    bert_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = bert_model.to(device)
    has_bert_model = True
    print("   - BERT (Multilingual): ✓")
except:
    bert_model = None
    bert_tokenizer = None
    has_bert_model = False
    print("   - BERT (Multilingual): ✗ (not trained yet)")
```

#### Değişiklik 2: BERT tahmin fonksiyonu ekle

**Ekle (satır ~130 civarı, `classify_ticket` fonksiyonunda):**
```python
# BERT model prediction (if available and selected)
if model_type == 'bert' and has_bert_model:
    inputs = bert_tokenizer(
        text,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        prediction_idx = probabilities.argmax().item()
        prediction = label_encoder.inverse_transform([prediction_idx])[0]
        confidence = float(probabilities[prediction_idx])
        
        # Top 3 predictions
        top_3_indices = probabilities.argsort(descending=True)[:3]
        top_3 = [
            {
                "label": label_encoder.inverse_transform([idx.item()])[0],
                "probability": float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
```

---

### Adım 4: Frontend'i Güncelle (5 dakika)

`src/templates/index.html` dosyasını aç ve model seçeneklerine BERT ekle:

**Bul (satır ~180 civarı):**
```html
<option value="lstm">LSTM Model</option>
<option value="ensemble">Ensemble Model</option>
```

**Ekle:**
```html
<option value="bert">BERT (Multilingual) 🌍</option>
```

---

### Adım 5: Test Et (5 dakika)

```bash
START_WEB_APP.bat
```

**Tarayıcıda test et:**
1. Model: "BERT (Multilingual)" seç
2. Türkçe ticket dene:
   - "Bilgisayarım açılmıyor ekran siyah kalıyor"
   - Beklenen: **Hardware** (~90% confidence)
3. İngilizce ticket dene:
   - "My laptop screen is broken"
   - Beklenen: **Hardware** (~92% confidence)

---

## 🎯 Beklenen Sonuçlar

### Performans:
- **İngilizce**: ~88-90% accuracy (önceki gibi)
- **Türkçe**: ~85-90% accuracy (yeni!)
- **Çok dilli model**: Her iki dili de anlıyor

### Test Senaryoları:

| Türkçe Ticket | Beklenen Kategori | Confidence |
|---------------|-------------------|------------|
| Bilgisayarım açılmıyor | Hardware | ~85-90% |
| Şifremi unuttum | Access | ~90-95% |
| İnternet çok yavaş | Network | ~85-90% |
| Word açılmıyor | Software | ~80-85% |
| İzin talebi nasıl yapılır | HR Support | ~75-85% |
| Yeni laptop siparişi | Purchase | ~80-90% |
| Disk doldu | Storage | ~85-90% |

---

## 💡 İpuçları

### Hata: "File not found: cleaned_data_multilingual.csv"
**Çözüm:** Adım 1'i yap (`python src/prepare_multilingual_data.py`)

### Hata: "BERT model not found"
**Çözüm:** Adım 2'yi tamamla (BERT eğitimi)

### GPU kullanımı
- Eğitim süresi RTX 2060 ile: ~1.5-2 saat
- CPU ile: ~8-10 saat (önerilmez)

### Model boyutu
- BERT modeli: ~700 MB
- Tokenizer: ~1 MB
- Toplam: ~701 MB

---

## 📊 Alternatif: Hızlı Test (Sadece LSTM)

Eğer BERT eğitimi çok uzun sürüyorsa, önce LSTM'i test edebilirsin:

### LSTM için Türkçe desteği:

1. **Veri setini güncelle:**
```bash
cd src
python prepare_multilingual_data.py
```

2. **LSTM'i yeniden eğit:**
```bash
jupyter notebook
# 02_word2vec_lstm.ipynb aç
# Veri yükleme satırını değiştir:
# df = pd.read_csv("../data/cleaned_data_multilingual.csv")
# Tüm cell'leri çalıştır
```

3. **Test et:**
```bash
START_WEB_APP.bat
# Model: "LSTM" seç
# Türkçe ticket dene
```

**Beklenen:** LSTM de Türkçe'yi öğrenecek (~70-80% accuracy)

---

## 🚀 Sonraki Adımlar (Opsiyonel)

### Daha Fazla Türkçe Veri Ekle:
- `data/turkish_tickets_professional.csv` dosyasına daha fazla örnek ekle
- Gerçek Türkçe ticket'larınız varsa onları kullanın
- Her kategoriye en az 50-100 Türkçe örnek ekleyin

### Fine-tuning:
- BERT'i Türkçe ile 1-2 epoch daha eğitin
- Learning rate'i düşürün (1e-5)
- Sadece Türkçe örneklerde fine-tune yapın

### Data Augmentation:
- Türkçe örnekleri paraphrase edin
- Eş anlamlı kelimeler kullanın
- Farklı yazım stilleri ekleyin

---

## ✅ Kontrol Listesi

- [ ] Adım 1: Veri seti oluşturuldu (`cleaned_data_multilingual.csv`)
- [ ] Adım 2: BERT modeli eğitildi (`bert_model.pt` kaydedildi)
- [ ] Adım 3: Web uygulaması güncellendi (BERT eklendi)
- [ ] Adım 4: Frontend güncellendi (BERT seçeneği eklendi)
- [ ] Adım 5: Test edildi (Türkçe + İngilizce)
- [ ] Bonus: LSTM de güncellendi (opsiyonel)

---

## 📝 Notlar

- **Orijinal veri korundu**: `cleaned_data.csv` değişmedi
- **Yeni veri**: `cleaned_data_multilingual.csv` oluşturuldu
- **Geriye dönük uyumluluk**: Eski modeller çalışmaya devam ediyor
- **80 Türkçe örnek**: Başlangıç için yeterli, daha fazla eklenebilir

---

**Son Güncelleme:** 21 Kasım 2025  
**Durum:** ✅ Hazır - Adımları Takip Et  
**Tahmini Süre:** 2-3 saat (eğitim dahil)

