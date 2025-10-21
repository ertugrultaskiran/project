# 🎓 BİTİRME PROJESİ - SUNUM ÖZETİ

**Öğrenci:** [Adınız]  
**Proje:** Müşteri Destek Taleplerinin Otomatik Sınıflandırılması  
**Tarih:** 21 Ekim 2025  
**Sonuç:** %88.82 Test Accuracy ✅

---

## 1️⃣ PROBLEM TANIMI

### Ne Sorunu Çözüyoruz?
Bir şirkette günde yüzlerce destek talebi geliyor. Bu talepleri doğru departmana yönlendirmek zaman alıyor ve hata yapılabiliyor.

### Çözümümüz:
Gelen talepleri otomatik olarak doğru kategoriye sınıflandıran bir yapay zeka sistemi.

---

## 2️⃣ VERİ SETİ

**Kaynak:** IT Destek Sistemi kayıtları  
**Boyut:** 47,837 ticket (destek talebi)  
**Kategoriler:** 8 farklı sınıf

### Kategoriler ve Örnekleri:

| Kategori | Örnek Talep | Adet |
|----------|-------------|------|
| Hardware | "Laptop screen broken" | 13,617 |
| HR Support | "Leave request approval" | 10,915 |
| Access | "Need VPN access" | 7,125 |
| Miscellaneous | "General question" | 7,060 |
| Storage | "Disk space needed" | 2,777 |
| Purchase | "Software license" | 2,464 |
| Internal Project | "Project resources" | 2,119 |
| Admin Rights | "Need admin access" | 1,760 |

**Veri Bölümü:** %80 Train, %10 Validation, %10 Test

---

## 3️⃣ KULLANILAN MODELLER

### Model 1: Baseline (TF-IDF + Logistic Regression)
**Ne:** Klasik makine öğrenmesi yaklaşımı  
**Nasıl:** Kelimelerin sıklığına göre sınıflandırma  
**Neden:** Hızlı referans metrik için  
**Sonuç:** %86.04

**Avantajları:**
- Çok hızlı (5 dk eğitim)
- Açıklanabilir
- Az kaynak gerekir

---

### Model 2: Deep Learning (Word2Vec + Bidirectional LSTM)
**Ne:** Derin öğrenme yaklaşımı  
**Nasıl:** Kelimeleri vektörlere çevirip, LSTM ile sıralı okuma  
**Neden:** Bağlam ve kelime sırasını anlamak için  
**Sonuç:** %87.00 (+0.96%)

**Avantajları:**
- Kelime sırasını anlar
- Anlamsal benzerlik yakalar
- Karmaşık kalıpları öğrenir

**Mimari:**
```
Input (Metin)
    ↓
Word2Vec Embedding (200 boyut)
    ↓
Bidirectional LSTM (128 units)
    ↓
Dense Layer (8 sınıf)
    ↓
Softmax (Olasılık dağılımı)
```

---

### Model 3: Ensemble (Model Kombinasyonu)
**Ne:** Birden fazla modelin birleşimi  
**Nasıl:** Baseline ve LSTM tahminlerinin ağırlıklı ortalaması  
**Neden:** Farklı modellerin güçlü yönlerini birleştirmek  
**Sonuç:** %88.40 (+2.36%)

**Avantajları:**
- Daha güvenilir
- Hata toleransı yüksek
- Kolay implement

**Yöntem:**
```
Ensemble = (0.5 × Baseline) + (0.5 × LSTM)
```

---

### Model 4: BERT Fine-Tuning (Transfer Learning)
**Ne:** Pre-trained transformer model  
**Nasıl:** Google'ın BERT modelini bizim veriye uyarlama  
**Neden:** En yüksek performans için state-of-the-art model  
**Sonuç:** %88.82 (+2.78%) 🏆 **EN İYİ**

**Avantajları:**
- Milyarlarca kelime ile önceden eğitilmiş
- Attention mechanism (dikkat mekanizması)
- Her kelimenin tüm bağlamı anlama
- Transfer learning ile az veri ile yüksek performans

**Teknik Detay:**
- Pre-trained: bert-base-uncased
- Fine-tuning: 3 epoch
- Optimizer: AdamW
- Learning rate: 2e-5
- GPU: NVIDIA RTX 2060

---

## 4️⃣ SONUÇLAR

### Performans Tablosu:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline | 86.04% | 87% | 86% | 86% |
| LSTM | 87.00% | 87% | 87% | 87% |
| Ensemble | 88.40% | 89% | 88% | 88% |
| **BERT** | **88.82%** | **89%** | **89%** | **89%** |

### Sınıf Bazında En İyi Performans (BERT):
- Purchase: F1 = 0.938 (Mükemmel!)
- Storage: F1 = 0.922
- Access: F1 = 0.915
- HR Support: F1 = 0.900

**Ortalama F1-Score:** 0.891 (Çok İyi!)

---

## 5️⃣ TEKNİK DETAYLAR

### Kullanılan Teknolojiler:
- **Python 3.13**
- **TensorFlow 2.20** (LSTM için)
- **PyTorch 2.9 + CUDA** (BERT için)
- **Transformers 4.57** (Hugging Face)
- **Scikit-learn, Pandas, NumPy**

### Veri İşleme:
- Text cleaning (URL, özel karakter temizleme)
- Tokenization
- Stratified split (sınıf dengesi)
- Class weights (dengesiz veri)

### Optimizasyonlar:
- Early stopping (overfitting önleme)
- Dropout regularization
- GPU acceleration
- Batch processing

---

## 6️⃣ SONUÇ VE DEĞERLENDİRME

### Başarılar:
✅ %88.82 test accuracy (hedef %85+ aşıldı)  
✅ 4 farklı yaklaşım karşılaştırıldı  
✅ Production-ready sistem geliştirildi  
✅ GPU optimizasyonu yapıldı  

### Öğrenilenler:
- Klasik ML vs Deep Learning farklılıkları
- Transfer learning'in gücü
- Ensemble yöntemlerinin faydası
- GPU kullanımının önemi

### Gelecek İyileştirmeler:
- Data augmentation ile %90+ accuracy
- Cross-validation ile daha güvenilir metrikler
- Daha fazla epoch ile BERT optimizasyonu
- Real-time deployment

---

## 7️⃣ DELİVERABLES (TESLİMATLAR)

### Kodlar:
- ✅ 4 farklı model implementasyonu
- ✅ Veri işleme pipeline'ı
- ✅ Evaluation scriptleri
- ✅ REST API

### Modeller:
- ✅ 4 eğitilmiş model dosyası
- ✅ Tokenizer'lar ve encoder'lar
- ✅ Model konfigürasyonları

### Raporlar:
- ✅ Confusion matrix
- ✅ Classification reports
- ✅ Model comparison charts
- ✅ Per-class performance analysis

### Dokümantasyon:
- ✅ README
- ✅ Kurulum rehberi
- ✅ API documentation
- ✅ Technical reports

---

## 🎤 HOCAYA 60 SANİYEDE ANLATIM

> **"Sayın Hocam,**
> 
> **Problemim:** 47 bin IT destek talebini otomatik kategorilere ayırmak.
> 
> **Çözümüm:** 4 farklı yaklaşım denedim:
> 
> 1. **Klasik ML** (TF-IDF + LogReg) → %86 - Hızlı baseline
> 2. **Derin Öğrenme** (Word2Vec + LSTM) → %87 - Bağlam anlama
> 3. **Ensemble** (Model birleştirme) → %88.4 - Güvenilirlik
> 4. **Transfer Learning** (BERT) → %88.82 - En iyi sonuç
> 
> **Teknik:**
> - GPU ile eğitim (NVIDIA RTX 2060)
> - Stratified train/test split
> - Class weights ile dengesiz veri çözümü
> - Production-ready REST API
> 
> **Sonuç:**
> - 8 kategoride %88.82 doğruluk
> - En iyi sınıflarda %94 F1-score
> - Sistemi gerçek zamanlı kullanıma hazır
> 
> Teşekkürler!"**

---

## 📊 SUNUM İÇİN GRAFİKLER

Hocaya göstermek için grafikler oluşturmamı ister misiniz?
- Model performans karşılaştırma grafiği
- Confusion matrix
- Training history grafikleri
- Sınıf bazında performans grafikleri

İster misiniz? 😊

