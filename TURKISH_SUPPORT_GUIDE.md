# 🇹🇷 Türkçe Ticket Desteği Ekleme Rehberi

## Durum
Web uygulaması çalışıyor ama modeller sadece İngilizce eğitilmiş.

## Çözüm Seçenekleri

### ✅ Seçenek 1: Hızlı Test (5 dakika)
Mevcut modelleri Türkçe ile test et. Veri setinde Türkçe örnekler varsa zaten çalışabilir.

**Test et:**
```
Web uygulamasında şunu dene:
"Bilgisayarım açılmıyor ve ekran yanmıyor"
```

Eğer mantıklı kategori döndürürse (Hardware), ek bir şey yapma!

---

### ✅ Seçenek 2: Çok Dilli BERT (2-3 saat, kalıcı çözüm)

#### Adım 1: Notebook'u güncelle

`src/03_bert_transformer.ipynb` dosyasını aç ve şu değişiklikleri yap:

**Değişiklik 1 - Tokenizer:**
```python
# ESKİ:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# YENİ:
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
```

**Değişiklik 2 - Model:**
```python
# ESKİ:
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', ...)

# YENİ:
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', ...)
```

#### Adım 2: Modeli yeniden eğit
```bash
cd src
jupyter notebook
# 03_bert_transformer.ipynb'i aç ve tüm cell'leri çalıştır
```

**Süre:** 1-2 saat (RTX 2060 ile)

#### Adım 3: Web uygulamasına BERT ekle (opsiyonel)

`src/web_app.py` içinde BERT modelini yükleyip kullanabilirsin.

---

### ✅ Seçenek 3: Türkçe Veri Ekle + Mevcut Modelleri Yeniden Eğit (4-5 saat)

#### Adım 1: Türkçe örnekler ekle
```bash
cd src
python add_turkish_data.py
```

Bu, `data/cleaned_data_with_turkish.csv` oluşturur (20 Türkçe örnek eklenmiş).

#### Adım 2: Notebookları güncelle

**Her notebook'ta veri yükleme satırını değiştir:**

```python
# ESKİ:
df = pd.read_csv("../data/cleaned_data.csv")

# YENİ:
df = pd.read_csv("../data/cleaned_data_with_turkish.csv")
```

**Güncellenecek dosyalar:**
- `src/01_baseline_tfidf_logreg.ipynb`
- `src/02_word2vec_lstm.ipynb`
- `src/03_bert_transformer.ipynb`
- `src/04_ensemble_model.py`

#### Adım 3: Modelleri yeniden eğit
```bash
jupyter notebook
# Notebookları sırayla çalıştır:
# 01 → 02 → 03
```

#### Adım 4: Ensemble'ı güncelle
```bash
cd src
python 04_ensemble_model.py
```

#### Adım 5: Web uygulamasını test et
```bash
START_WEB_APP.bat
```

---

## 🎯 Önerilen Yaklaşım

**Hemen şimdi:**
1. Web uygulamasında Türkçe ticket dene (belki çalışıyordur)

**Zaman varsa (1-2 gün):**
2. Çok dilli BERT kullan (`bert-base-multilingual-cased`)
3. Modelleri yeniden eğit
4. Test et ve raporları güncelle

**Alternatif (daha hızlı):**
- Sadece LSTM modelini Türkçe verilerle fine-tune et
- Word2Vec Türkçe kelimeler öğrenecektir

---

## 📊 Beklenen Sonuçlar

| Yaklaşım | Süre | Türkçe Performans | İngilizce Performans |
|----------|------|-------------------|---------------------|
| Test et (şu anki) | 5 dk | Belirsiz | ✅ 88% |
| Çok dilli BERT | 2-3 saat | ✅ ~85-90% | ✅ ~88-90% |
| Türkçe veri ekle | 4-5 saat | ✅ ~80-85% | ✅ 88% |

---

## 💡 Pratik İpuçları

### Türkçe karakter sorunları için:
```python
# utils.py'deki basic_clean fonksiyonunu güncelle:
def basic_clean(s: str) -> str:
    s = s.lower()
    # Türkçe karakterleri koru
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    # Türkçe için: [a-zçğıöşü] kullan
    s = re.sub(r"[^a-zçğıöşü0-9\-'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
```

### Test için örnek Türkçe ticket'lar:
- "Bilgisayarım açılmıyor ve ekran yanmıyor" → Hardware
- "Şifremi unuttum ve sisteme giriş yapamıyorum" → Access
- "VPN bağlantısı kopuyor sürekli" → Network
- "Yazılım güncellemesi nasıl yapılır" → Software

---

## ❓ Soru-Cevap

**S: Şu anki modeller Türkçe'yi anlıyor mu?**
C: Test etmen lazım. Word2Vec ve LSTM kelime bazlı çalıştığı için veri setinde Türkçe varsa kısmen çalışabilir.

**S: En hızlı çözüm nedir?**
C: Çok dilli BERT kullanmak (2-3 saat).

**S: Hem Türkçe hem İngilizce çalışsın istiyorum**
C: `bert-base-multilingual-cased` kullan, 100+ dili destekler.

**S: Sunum için yetişir mi?**
C: Eğer 1-2 gün vaktiniz varsa evet. Yoksa "Türkçe desteği opsiyonel özellik, çok dilli model kullanılarak eklenebilir" de.

---

**Güncelleme:** 21 Kasım 2025
**Durum:** Rehber hazır, uygulama bekliyor

