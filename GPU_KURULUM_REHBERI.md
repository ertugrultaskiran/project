# 🚀 GPU (CUDA) Kurulum Rehberi - Windows

## Durum: GPU var ama Python göremiyormuş ⚠️

**Çözüm:** CUDA Toolkit + PyTorch GPU versiyonu kurulumu

---

## 📋 Gerekli Kurulumlar:

### 1️⃣ CUDA Toolkit (NVIDIA Driver)

#### Adım 1: NVIDIA GPU'nuz Hangi Model?
Windows'ta kontrol edin:
- **Ctrl + Shift + Esc** → Görev Yöneticisi
- **Performans** sekmesi
- **GPU** → Model adını görün (örn: RTX 3060, GTX 1660 Ti)

#### Adım 2: CUDA İndirin
1. https://developer.nvidia.com/cuda-downloads adresine gidin
2. Seçimler:
   - **Operating System:** Windows
   - **Architecture:** x86_64
   - **Version:** 10 veya 11 (Windows sürümünüz)
   - **Installer Type:** exe (local) - Önerilen
3. **Download** → ~3 GB indirecek

#### Adım 3: CUDA Kurun
1. İndirilen dosyayı çalıştırın (yönetici olarak)
2. **Express (Recommended)** seçin
3. Next → Next → Install
4. ⏱️ Süre: ~10-15 dakika
5. ⚠️ **ÖNEMLİ:** Bilgisayarı yeniden başlatın!

---

### 2️⃣ PyTorch GPU Versiyonu

CUDA kurduktan sonra:

#### Kolay Yol: Hazır Script (Önerilen)
Proje klasöründe `install_pytorch_gpu.bat` dosyası oluşturuldu.

**Çalıştır:**
1. Dosya Gezgini'nde proje klasörünü açın
2. `install_pytorch_gpu.bat` dosyasına çift tıklayın
3. Bekleyin (~5-10 dakika)

#### Manual Yol: Komut Satırı
```bash
# 1. Mevcut PyTorch'u kaldır
pip uninstall torch torchvision torchaudio -y

# 2. GPU versiyonunu kur (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**VEYA CUDA 12.x için:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ✅ Kontrol: GPU Çalışıyor mu?

Kurulum sonrası Jupyter'da test edin:

```python
import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("✅ HAZIR!")
else:
    print("❌ Hala sorun var")
```

**Beklenen Çıktı:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 (veya sizin GPU'nuz)
✅ HAZIR!
```

---

## 🎯 Hızlı Kurulum Özeti:

1. **CUDA Toolkit İndir ve Kur** (~15 dk)
   - https://developer.nvidia.com/cuda-downloads
   - Express installation
   - Bilgisayarı yeniden başlat

2. **PyTorch GPU Kur** (~10 dk)
   - `install_pytorch_gpu.bat` dosyasına çift tık
   - Veya: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

3. **Test Et**
   - Jupyter'da yukarıdaki kodu çalıştır
   - True görmeli

**Toplam Süre:** ~30 dakika (indirme hızına bağlı)

---

## 🆘 Sorun mu Yaşıyorsunuz?

### Sorun 1: "CUDA hala False"
**Çözüm:**
- Bilgisayarı yeniden başlattınız mı?
- NVIDIA Driver güncel mi? → https://www.nvidia.com/Download/index.aspx
- `nvidia-smi` komutunu çalıştırın, GPU görünüyor mu?

### Sorun 2: "CUDA versiyonu uyumsuz"
**Çözüm:**
```bash
# CUDA versiyonunuzu öğrenin:
nvidia-smi

# Çıktıda "CUDA Version: 12.x" gibi bir şey göreceksiniz
# Ona göre PyTorch kurun:
# CUDA 11.8 için: cu118
# CUDA 12.x için: cu121
```

### Sorun 3: "İndirme çok yavaş"
**Alternatif:** Google Colab kullanın
- Ücretsiz GPU
- Kurulum gerektirmez
- Hemen başlayabilirsiniz

---

## 🚀 Alternatif: Google Colab (Kurulum İstemiyorsanız)

GPU kurulumu istemiyorsanız veya sorun yaşıyorsanız:

### Google Colab ile BERT Eğitimi:

1. **Colab'a git:** https://colab.research.google.com

2. **Notebook yükle:**
   - File → Upload notebook
   - `src/03_bert_transformer.ipynb` seçin

3. **GPU Aktif et:**
   - Runtime → Change runtime type
   - Hardware accelerator → **GPU**
   - Save

4. **Veriyi yükle:**
   ```python
   from google.colab import files
   uploaded = files.upload()  # cleaned_data.csv yükle
   ```

5. **Çalıştır:**
   - Runtime → Run all
   - Bekle (~2-3 saat)

6. **Model'i indir:**
   ```python
   from google.colab import files
   files.download('models/bert_model.pt')
   ```

---

## 📊 Karşılaştırma:

| Seçenek | Kurulum | Süre | Maliyet |
|---------|---------|------|---------|
| **Kendi GPU'nuz** | ~30 dk | 2-3 saat | Ücretsiz |
| **Google Colab** | 0 dk | 2-3 saat | Ücretsiz |
| **CPU** | 0 dk | 8-12 saat | Ücretsiz |

---

## 💡 Önerim:

1. **Zamanınız varsa:** CUDA + PyTorch GPU kurun
   - Tek seferlik kurulum
   - Sonra hep hızlı olacak
   - Diğer projeler için de kullanabilirsiniz

2. **Hemen başlamak istiyorsanız:** Google Colab
   - Şimdi başlayın
   - CUDA kurulumunu sonra yapın

---

## ✅ Kurulum Sonrası:

BERT eğitimine başlayın:

```bash
# Python scripti ile:
python src/train_bert.py

# Veya Jupyter'da:
# Tüm hücreleri sırayla çalıştırın
```

**Beklenen Süre:** ~2-3 saat  
**Beklenen Accuracy:** %90-93

---

**Hangisini tercih ediyorsunuz?**
1. CUDA kurulumu yapmak (30 dk + her zaman hızlı)
2. Google Colab kullanmak (hemen başla, kurulum yok)

Söyleyin size yardımcı olayım! 🚀


