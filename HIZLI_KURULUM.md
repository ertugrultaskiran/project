# ⚡ PyTorch GPU Hızlı Kurulum

## 🎯 CUDA Kurulumu Bittikten Sonra:

### Adım 1: Bilgisayarı Yeniden Başlat ⚠️
**MUTLAKA** yeniden başlatın!

### Adım 2: PowerShell veya CMD Açın

**2 Seçenek:**

**A) PowerShell (Önerilen):**
- Windows tuşu + X
- "Windows PowerShell" seçin

**B) CMD:**
- Windows tuşu + R
- "cmd" yazın
- Enter

### Adım 3: Proje Klasörüne Git

```powershell
cd C:\Users\ertug\OneDrive\Masaüstü\project
```

### Adım 4: PyTorch GPU Kur

**Tek komut - Kopyalayıp yapıştırın:**

```powershell
pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Bekleyin (~5-10 dakika)...

### Adım 5: Test Et

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Görmek istediğiniz:**
```
CUDA: True
```

---

## ✅ Başarılı Olduysa:

BERT eğitimine başlayın:

```powershell
python src/train_bert.py
```

---

## ❌ False Görüyorsanız:

1. Bilgisayarı yeniden başlattınız mı? ⚠️
2. NVIDIA driver güncel mi?
   ```powershell
   nvidia-smi
   ```
3. Farklı CUDA versiyonu deneyin:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 🚀 Ya da Google Colab:

Kurulum istemiyorsanız:
1. https://colab.research.google.com
2. Dosyalarınızı yükleyin
3. Hemen eğitime başlayın!

---

**Kurulum bitince bana haber verin!** 😊

