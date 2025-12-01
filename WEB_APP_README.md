# 🌐 IT Ticket Classification - Web Application

## 🎯 Özellikler

Modern ve kullanıcı dostu web uygulaması ile IT destek taleplerini **gerçek zamanlı** sınıflandırın!

### ✨ Ana Özellikler:

1. **📊 Interactive Dashboard**
   - Toplam sınıflandırılmış ticket sayısı
   - Ortalama confidence skoru
   - En sık kategori
   - Aktif model sayısı

2. **💬 Chatbot Widget (Sağ Alt)**
   - Müşteri talebini yaz
   - Otomatik kategorize et
   - Confidence skorunu gör
   - Top 3 tahminleri görüntüle

3. **📋 Classification History Panel**
   - Tüm sınıflandırmaları listele
   - Kategori badge'leri
   - Confidence bar'ları
   - Zaman damgaları

4. **🤖 Multiple Model Support**
   - Baseline Model (TF-IDF + LogReg)
   - LSTM Model (Word2Vec + BiLSTM)
   - Ensemble Model (Kombinasyon)
   - Custom Attention Model (Sizin modeliniz!)

---

## 🚀 Hızlı Başlangıç

### Adım 1: Gerekli Paketleri Yükle

```bash
pip install Flask Flask-Cors
```

veya

```bash
pip install -r requirements.txt
```

### Adım 2: Web Uygulamasını Başlat

**Windows:**
```bash
START_WEB_APP.bat
```

**veya Manuel:**
```bash
cd src
python web_app.py
```

### Adım 3: Tarayıcıda Aç

```
http://localhost:5000
```

---

## 📱 Kullanım

### 1. Chatbot ile Ticket Sınıflandırma

1. Sağ alttaki **chatbot widget**'ı görün
2. Model seçin (Ensemble önerilir)
3. Müşteri talebini yazın:
   ```
   "I need urgent access to SAP system"
   ```
4. **Classify** butonuna tıklayın
5. Sonucu görün:
   - Kategori (ör. Access)
   - Confidence (ör. 95.2%)
   - Top 3 tahmin

### 2. Panel'de Görüntüleme

- Sınıflandırılan ticket otomatik olarak **panel**'e eklenir
- Tüm geçmişi görebilirsiniz
- Her ticket için:
  - ID
  - Açıklama
  - Kategori (renkli badge)
  - Confidence (progress bar)
  - Model
  - Zaman

### 3. İstatistikler

Üstteki kartlar otomatik güncellenir:
- 📊 Toplam Ticket
- 🧠 Aktif Model Sayısı
- 📈 Ortalama Confidence
- 🏷️ En Sık Kategori

---

## 🎨 Ekran Görüntüleri

### Ana Dashboard
```
┌────────────────────────────────────────────────┐
│  🤖 IT Ticket Classifier                      │
├────────────────────────────────────────────────┤
│  📊 Stats Cards (4 adet)                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ 15   │ │  3   │ │ 87%  │ │Access│        │
│  └──────┘ └──────┘ └──────┘ └──────┘        │
├────────────────────────────────────────────────┤
│  📋 Classification History Table               │
│  ┌─────────────────────────────────────────┐  │
│  │ ID  │ Text │ Category │ Confidence...  │  │
│  ├─────────────────────────────────────────┤  │
│  │ #ab12│ Need│  Access  │ ████████ 95%   │  │
│  └─────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
                                    ┌─────────────┐
                                    │ 💬 Chatbot  │
                                    │             │
                                    │ [Model]     │
                                    │ [Text...]   │
                                    │ [Classify]  │
                                    └─────────────┘
```

---

## 🔧 Teknik Detaylar

### Backend (Flask)

**Endpoints:**
- `GET /` - Ana dashboard page
- `POST /api/classify` - Ticket sınıflandır
- `GET /api/tickets` - Tüm ticket'ları getir
- `GET /api/stats` - İstatistikleri getir
- `GET /api/health` - Health check

**Dosya:** `src/web_app.py`

### Frontend

**Teknolojiler:**
- HTML5
- CSS3 (Modern, responsive)
- Vanilla JavaScript (No framework)
- Font Awesome icons
- Google Fonts (Inter)

**Dosyalar:**
- `src/templates/index.html` - Ana sayfa
- `src/static/css/style.css` - Stiller
- `src/static/js/app.js` - JavaScript logic

---

## 🎯 HOCAYA SUNUM İÇİN

### Neden Bu Önemli?

1. **Canlı Demo** 🎥
   - Hocaya gerçek zamanlı gösterebilirsiniz
   - "İşte çalışan sisteminiz" diyebilirsiniz
   - Interactive ve etkileyici

2. **Production-Ready** 🚀
   - Gerçek kullanıma hazır
   - Modern UI/UX
   - Professional görünüm

3. **Tüm Modelleri Test Et** 🧪
   - 4 farklı model karşılaştırma
   - Hangi model daha iyi?
   - Ensemble'ın gücünü göster

4. **User Experience** 👥
   - Chatbot interface (kullanıcı dostu)
   - Anlaşılır görselleştirme
   - Real-world application

### Demo Senaryosu (Hocaya)

```
1. Tarayıcıyı aç
   → http://localhost:5000

2. Dashboard'u göster
   → "Burada tüm istatistikleri görüyoruz"

3. Chatbot'u aç
   → "Kullanıcı buradan ticket giriyor"

4. Örnek ticket gir:
   "I need urgent access to SAP system. My password is not working."

5. Classify'a tıkla
   → "Model tahmin yapıyor..."

6. Sonucu göster:
   → "Category: Access (95.2% confidence)"
   → "Top 3 predictions de burada"

7. Panel'i göster
   → "Ticket otomatik panele düştü"
   → "Tüm geçmişi burada görebiliyoruz"

8. Farklı modelleri dene
   → "Baseline, LSTM, Ensemble karşılaştırması"
```

---

## 🛠️ Özelleştirme

### Renkler Değiştir

`src/static/css/style.css` dosyasında:

```css
:root {
    --primary-color: #2563eb;  /* Ana renk */
    --secondary-color: #10b981; /* İkincil renk */
    /* ... diğer renkler ... */
}
```

### Kategori Renkleri

Her kategorinin kendine özel badge rengi var:
- Access → Mavi
- Hardware → Sarı
- HR Support → Yeşil
- Admin Rights → Mor
- vb.

---

## 📊 API Kullanımı (Opsiyonel)

Dışarıdan da kullanabilirsiniz:

```bash
# Ticket sınıflandır
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I need access to database", "model": "ensemble"}'

# Tüm ticket'ları getir
curl http://localhost:5000/api/tickets

# İstatistikler
curl http://localhost:5000/api/stats
```

---

## ❓ Sorun Giderme

### Hata: "Module flask not found"
```bash
pip install Flask Flask-Cors
```

### Hata: "Models not found"
Önce modelleri eğitin:
```bash
jupyter notebook src/01_baseline_tfidf_logreg.ipynb
jupyter notebook src/02_word2vec_lstm.ipynb
```

### Hata: "Port 5000 already in use"
Başka port kullanın:
```python
# src/web_app.py'de:
app.run(host='0.0.0.0', port=5001, debug=True)
```

### Chatbot görünmüyor
Tarayıcı console'u kontrol edin (F12):
- JavaScript hataları var mı?
- Network istekleri başarılı mı?

---

## 🎓 Akademik Değer

Bu web uygulaması projenize şu değeri katar:

1. ✅ **Practical Implementation**
   - Teoriden pratiğe
   - Gerçek kullanım senaryosu

2. ✅ **User-Centric Design**
   - UX/UI düşünülmüş
   - Accessibility

3. ✅ **Production Readiness**
   - API endpoints
   - Error handling
   - Scalability considerations

4. ✅ **Interactive Demonstration**
   - Hocaya canlı gösterim
   - Etkileyici sunum

---

## 📝 Notlar

- Web app **in-memory** çalışır (tickets_history)
- Production için **database** kullanın (SQLite, PostgreSQL)
- Güvenlik için **authentication** ekleyin
- Deployment için **Docker** kullanın

---

## 🚀 Sonraki Adımlar

1. **Gelişmiş Özellikler:**
   - Ticket export (CSV/Excel)
   - Filtering ve sorting
   - Bulk classification
   - User authentication

2. **Visualization:**
   - Grafik ekleme (Chart.js)
   - Trend analysis
   - Category distribution pie chart

3. **Deployment:**
   - Heroku'ya deploy
   - Docker containerization
   - Cloud hosting (AWS, Azure)

---

## 📞 Destek

Sorunuz var mı?
- Backend hatası: `src/web_app.py`'yi kontrol edin
- Frontend hatası: Browser console'u kontrol edin
- Model hatası: Modellerin eğitildiğinden emin olun

---

**Hazırlayan:** AI Assistant  
**Tarih:** 16 Kasım 2025  
**Durum:** ✅ Kullanıma hazır

**🎉 Hocaya göstermek için mükemmel bir canlı demo!**

