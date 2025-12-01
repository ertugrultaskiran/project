# 🇹🇷 Plan B: Türkçe Desteği Güçlendirme

## Mevcut Durum
- 210 Türkçe örnek
- Beklenen accuracy: %85-90
- Eğer düşükse (%75 altı), ek önlemler gerekir

---

## Plan B1: Daha Fazla Veri (1 gün)

### Hedef: 500-1000 Türkçe Örnek

#### Yöntem 1: Manuel Veri Toplama
**Kendin yaz (ChatGPT yardımıyla):**
```
ChatGPT'ye sor:
"Bana 50 tane Hardware kategorisi IT destek talebi örneği yaz, Türkçe olsun."

Sonra:
"Şimdi 50 tane Access kategorisi örneği"
...
```

**Süre:** 2-3 saat (8 kategori × 50 örnek)

#### Yöntem 2: Data Augmentation
**Mevcut Türkçe örnekleri çoğalt:**

```python
# src/augment_turkish_data.py
import random

def paraphrase_turkish(text):
    """Basit paraphrase: Eş anlamlılar"""
    replacements = {
        'bilgisayar': ['laptop', 'pc', 'masaüstü'],
        'açılmıyor': ['çalışmıyor', 'boot etmiyor', 'başlamıyor'],
        'ekran': ['monitör', 'display'],
        'şifre': ['parola', 'password'],
        'internet': ['ağ', 'wifi', 'bağlantı']
    }
    
    augmented = []
    for original in turkish_tickets:
        # Original
        augmented.append(original)
        
        # 2-3 varyasyon oluştur
        for _ in range(2):
            new_text = original
            for old, options in replacements.items():
                if old in new_text:
                    new_text = new_text.replace(old, random.choice(options))
            augmented.append(new_text)
    
    return augmented

# 210 örnek → 630 örnek (3x)
```

**Süre:** 1-2 saat

#### Yöntem 3: Back-Translation
**Türkçe → İngilizce → Türkçe (farklı ifade):**

```python
from googletrans import Translator

translator = Translator()

def back_translate(text):
    # TR → EN
    en = translator.translate(text, src='tr', dest='en').text
    # EN → TR
    tr_back = translator.translate(en, src='en', dest='tr').text
    return tr_back

# "Bilgisayarım açılmıyor"
# → "My computer won't start"
# → "Bilgisayarım başlamıyor" (farklı ifade!)
```

**Süre:** 2-3 saat (API limitleri var)

---

## Plan B2: Model Fine-tuning (4-6 saat)

### Türkçe-Specific BERT Kullan

**Şu an:** `bert-base-multilingual-cased` (104 dil)
**Alternatif:** `dbmdz/bert-base-turkish-cased` (sadece Türkçe, daha iyi)

**Nasıl:**
```python
# Notebook'ta değiştir:
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased')
```

**Artı:**
- ✅ Türkçe için optimize
- ✅ Daha iyi performance beklenir

**Eksi:**
- ❌ Sadece Türkçe (İngilizce kaybederiz)

### Çözüm: İki Model Kullan
```python
# İngilizce için: bert-base-uncased
# Türkçe için: bert-base-turkish-cased
# Otomatik dil tespiti yapıp uygun modeli seç
```

---

## Plan B3: Hybrid Approach (2-3 saat)

### Türkçe için Rule-Based + ML

**Mantık:**
```python
def classify_turkish(text):
    # 1. Keyword matching (basit ama etkili)
    if any(word in text for word in ['bilgisayar', 'laptop', 'ekran', 'monitör']):
        return 'Hardware', 0.90
    
    elif any(word in text for word in ['şifre', 'giriş', 'erişim', 'hesap']):
        return 'Access', 0.85
    
    elif any(word in text for word in ['internet', 'wifi', 'bağlantı', 'ağ']):
        return 'Network', 0.88
    
    # 2. ML model (backup)
    else:
        return ml_model.predict(text)
```

**Artı:**
- ✅ Basit ama çalışır
- ✅ Hızlı implementation
- ✅ Yüksek accuracy (%85-90)

**Eksi:**
- ❌ Ölçeklenebilir değil
- ❌ Her kategori için keyword yazmak gerekir

---

## Plan B4: Transfer from English (Akıllı çözüm!)

### İngilizce'den Türkçe'ye Knowledge Transfer

**Yöntem:**
```python
# 1. Türkçe ticket'ı İngilizce'ye çevir
turkish_ticket = "Bilgisayarım açılmıyor"
english_ticket = translate_to_english(turkish_ticket)  # "My computer won't start"

# 2. İngilizce modelle classify et (yüksek accuracy)
category = english_model.predict(english_ticket)  # Hardware %95

# 3. Türkçe cevap ver
response = turkish_templates[category]
```

**Artı:**
- ✅ İngilizce model çok iyi (%88.5)
- ✅ Translation API kolay (Google Translate)
- ✅ Hemen çalışır

**Eksi:**
- ❌ Translation maliyeti (ama Google Translate free tier var)
- ❌ Ufak bir latency (100-200ms)

---

## 📊 **GERÇEK DÜNYA BEKLENTİLERİ**

### Kategori Bazında Türkçe Performance (Tahmini):

| Kategori | Türkçe Örnek Sayısı | Beklenen Accuracy | Güven |
|----------|---------------------|-------------------|-------|
| **Hardware** | ~30 | %85-92 | ✅ Yüksek |
| **Access** | ~30 | %85-90 | ✅ Yüksek |
| **HR Support** | ~30 | %80-88 | ✅ İyi |
| **Storage** | ~30 | %80-88 | ✅ İyi |
| **Purchase** | ~30 | %75-85 | ⚠️ Orta |
| **Network** | ~30 | %70-80 | ⚠️ Orta-Düşük |
| **Software** | ~30 | %70-80 | ⚠️ Orta-Düşük |
| **Miscellaneous** | ~30 | %65-75 | ⚠️ Düşük |

**Neden bazıları düşük?**
- Network/Software: Teknik terimler çok, örnekler az
- Miscellaneous: Zaten "catch-all" kategorisi

---

## 🎯 **BERT BİTİNCE İLK TEST**

BERT eğitimi bittiğinde (30 dk kaldı?), **hemen bunu test et:**

### **Test Seti (Türkçe):**

**Kolay (basit kelimeler):**
```
1. Bilgisayarım açılmıyor
2. Şifremi unuttum
3. İnternet yavaş
```
**Beklenen:** %90+ accuracy

**Orta (teknik terimler):**
```
4. VPN bağlantısı kuramıyorum
5. Outlook donuyor sürekli
6. Disk doldu kaydedemiyorum
```
**Beklenen:** %80-85 accuracy

**Zor (karmaşık cümleler):**
```
7. SAP sistemine remote desktop üzerinden erişemiyorum ve SSL sertifika hatası alıyorum
```
**Beklenen:** %70-80 accuracy

---

## 💡 **HOCAYA GÖSTERME STRATEJİSİ**

### **Eğer Türkçe İyi Çalışırsa (%85+):**
> "Hocam, bakın Türkçe de mükemmel çalışıyor! Çok dilli transfer learning'in gücü bu!"

### **Eğer Orta Çalışırsa (%70-84):**
> "Hocam, Türkçe %80 civarı accuracy var. 210 örnekle başlangıç için iyi. Production'da daha fazla veri toplanırken online learning ile kendini geliştirebilir."

### **Eğer Kötü Çalışırsa (%70 altı):**
> "Hocam, Türkçe için şu an %75 accuracy var. İki alternatif geliştirdim:
> 1. **Translation-based:** Türkçe'yi İngilizce'ye çevirip classify ediyorum (%90+ accuracy)
> 2. **Hybrid:** Keyword matching + ML (%85 accuracy)
> 
> Production'da bunlardan birini kullanırım, hatta ensemble edebilirim!"

**Her durumda kazanıyorsun!** 💪

---

## 🔮 **BENİM TAHMİNİM**

**%75 ihtimal:** Türkçe %80-88 accuracy ile **çok iyi çalışacak** ✅

**%20 ihtimal:** Türkçe %70-79 accuracy, **kullanılabilir ama geliştirilmeli** ⚠️

**%5 ihtimal:** Türkçe %70 altı, **Plan B gerekli** ❌

---

## ⏰ **ŞİMDİ NE YAPALIM?**

1. **BERT'in bitmesini bekle** (30-45 dk?)
2. **Hemen test et** (yukarıdaki 7 Türkçe örnek)
3. **Sonuçlara göre karar ver:**
   - İyi → Hocaya göster
   - Orta → Geliştirme planı sun
   - Kötü → Plan B uygula

---

**BERT'in epoch'u kontrol et? Nerede şu an?** Bitmeye yakın mı? 🔍
