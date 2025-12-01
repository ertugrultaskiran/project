# 🧪 Web App Test Rehberi

## Hızlı Test (5 dakika)

### 1. Flask'ı Yükle (1 dakika)

```bash
pip install Flask Flask-Cors
```

### 2. Web App'i Başlat (30 saniye)

**Windows:**
```bash
START_WEB_APP.bat
```

**veya:**
```bash
cd src
python web_app.py
```

### 3. Tarayıcıda Aç (10 saniye)

```
http://localhost:5000
```

### 4. Test Et (3 dakika)

#### Test 1: Dashboard Açıldı mı?
- ✅ 4 stat kartı görünüyor mu?
- ✅ Tablo görünüyor mu?
- ✅ Chatbot sağ altta mı?

#### Test 2: Chatbot Çalışıyor mu?
1. Chatbot'a tıkla (açıldı mı?)
2. Model seç: "Ensemble Model"
3. Şunu yaz:
   ```
   I need urgent access to SAP system
   ```
4. "Classify" butonuna tıkla
5. Bekle...
6. Sonuç geldi mi?
   - ✅ Kategori göründü mü?
   - ✅ Confidence % göründü mü?
   - ✅ Top 3 predictions var mı?

#### Test 3: Panel Güncellendi mi?
1. Yukarı scroll yap
2. Tabloya bak
3. ✅ Yeni ticket eklendi mi?
4. ✅ Badge rengi var mı?
5. ✅ Confidence bar dolu mu?

#### Test 4: Stats Güncellendi mi?
1. En üstteki kartlara bak
2. ✅ "Total Tickets" = 1 mi?
3. ✅ "Avg. Confidence" > 0 mı?
4. ✅ "Most Common Category" = Access mi?

---

## Beklenen Sonuçlar

### Test Ticket:
```
"I need urgent access to SAP system"
```

### Beklenen Çıktı:
- **Category:** Access
- **Confidence:** ~85-95%
- **Top 3:**
  1. Access: ~90%
  2. HR Support: ~5%
  3. Hardware: ~3%

---

## Hata Durumunda

### Hata: "Connection refused"
**Çözüm:** Web app çalışıyor mu kontrol et
```bash
# Yeni terminal aç
cd src
python web_app.py
```

### Hata: "Models not found"
**Çözüm:** Modelleri eğit
```bash
jupyter notebook src/01_baseline_tfidf_logreg.ipynb
jupyter notebook src/02_word2vec_lstm.ipynb
```

### Hata: "500 Internal Server Error"
**Çözüm:** Terminal'de hatayı gör
- Flask çıktısında hata var mı?
- Model dosyaları `models/` klasöründe mi?

### Chatbot açılmıyor
**Çözüm:** Browser console kontrol et (F12)
- JavaScript hataları var mı?
- CSS yüklendi mi?

---

## HOCAYA GÖSTERME CHECKLİSTİ

### Sunum Öncesi (5 dakika önce):

- [ ] Web app çalışıyor mu?
  ```bash
  http://localhost:5000
  ```

- [ ] Tarayıcı tam ekran mı?

- [ ] Test ticket'ı hazırladın mı?
  ```
  "I need urgent access to SAP system. Password reset required."
  ```

- [ ] İnternet bağlantısı var mı? (Font ve ikon için)

- [ ] Backup plan hazır mı? (Screenshot'lar)

---

## Demo Senaryosu (2 dakika)

### Adım 1: Dashboard'u Göster (15 saniye)
> "Hocam, bu bizim web arayüzümüz. Burada gerçek zamanlı sınıflandırma yapabiliyoruz."

### Adım 2: Chatbot'u Aç (10 saniye)
> "Kullanıcı sağ alttaki chatbot'tan talebini giriyor."

### Adım 3: Örnek Ticket (20 saniye)
Model: Ensemble
Text: "I need urgent access to SAP system. Password reset required."
> "Örnek bir müşteri talebi giriyorum..."

### Adım 4: Classify (30 saniye)
[Classify butonuna tıkla]
> "Model tahmin yapıyor... İşte sonuç!"

### Adım 5: Sonucu Göster (30 saniye)
> "Kategori: Access, Confidence: %92.5"
> "Top 3 tahmini de görebiliyoruz."
> "Panel'e otomatik eklendi."

### Adım 6: Panel (15 saniye)
[Yukarı scroll]
> "Tüm geçmiş burada. Her ticket için detaylı bilgi var."

**TOPLAM: 2 dakika** ✅

---

## Ekstra Demo Fikirleri

### Farklı Kategoriler Göster:

1. **Hardware:**
   ```
   "My laptop screen is broken and needs replacement"
   ```
   → Category: Hardware (~90%)

2. **HR Support:**
   ```
   "I need to submit my leave request for next week"
   ```
   → Category: HR Support (~88%)

3. **Storage:**
   ```
   "My OneDrive is full, need more storage space"
   ```
   → Category: Storage (~85%)

### Model Karşılaştırması:

Aynı ticket'ı farklı modellerle dene:
1. Baseline Model
2. LSTM Model
3. Ensemble Model
4. Custom Attention

> "Bakın, ensemble en iyi sonucu veriyor!"

---

## Sorulan Sorular ve Cevaplar

### S: "Bu gerçek zamanlı mı?"
**C:** "Evet hocam, backend Flask API çalışıyor. Her ticket anında sınıflandırılıyor."

### S: "Hangi modeli kullanıyor?"
**C:** "Kullanıcı seçebiliyor. Baseline, LSTM, Ensemble veya Custom Attention modelimiz. Ensemble önerilir çünkü en iyi sonucu veriyor."

### S: "Veri kayıtlanıyor mu?"
**C:** "Şu anda in-memory, ama production'da database kullanılabilir. SQLite veya PostgreSQL eklenebilir."

### S: "Mobilde çalışır mı?"
**C:** "Evet hocam, responsive design. Telefon ve tablette de çalışır."

### S: "API var mı?"
**C:** "Evet! REST API endpoints var. Dışarıdan da kullanılabilir. Curl ile veya başka uygulamalardan."

---

## 🎯 Başarı Kriterleri

Web app gösteriminiz başarılı sayılır eğer:

1. ✅ Hocaya canlı demo yaptınız
2. ✅ En az 3 farklı ticket sınıflandırdınız
3. ✅ Sonuçlar panele düştü
4. ✅ Confidence skorları göründü
5. ✅ Hoca UI/UX'den etkilendi
6. ✅ "Production-ready" izlenimi bıraktı

---

## 💡 Pro Tips

1. **Hazırlık:** Önceden 2-3 test ticket gir, paneli dolu göster
2. **Backup:** Screenshot'lar al (hata olursa göster)
3. **Anlatım:** Teknik detaya boğma, sonuca odaklan
4. **Özgüven:** "Bu working prototype" de
5. **Vurgu:** "Sadece model değil, full-stack application geliştirdim"

---

## 📸 Screenshot Checklist (Backup)

Hata durumunda göstermek için:

- [ ] Dashboard (boş)
- [ ] Dashboard (dolu)
- [ ] Chatbot (açık)
- [ ] Classification result
- [ ] Ticket in panel
- [ ] Stats updated

---

**🎉 HAZIRSSINIZ!**

Web app çalışıyor ve demo yapabilirsiniz.

**Son kontrol:**
```bash
cd src
python web_app.py
```

**Tarayıcıda:**
```
http://localhost:5000
```

**Başarılar! 🚀**

