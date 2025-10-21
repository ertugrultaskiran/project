# 🚀 PyTorch GPU Kurulum Rehberi

## ⏱️ Ne Zaman Yapılmalı?

1. ✅ CUDA Toolkit kurulumu bitti
2. ✅ Bilgisayar yeniden başlatıldı
3. ✅ Şimdi PyTorch GPU kurulumuna geçilecek

---

## 🎯 Kurulum Yöntemleri

### Yöntem 1: Otomatik Script (Önerilen - En Kolay)

1. **Dosya Gezgini'ni Açın**
   - Windows tuşu + E

2. **Proje Klasörüne Gidin**
   - `C:\Users\ertug\OneDrive\Masaüstü\project`

3. **Script'i Çalıştırın**
   - `install_pytorch_gpu.bat` dosyasına **çift tıklayın**
   - Siyah pencere açılacak
   - İşlem otomatik yapılacak (~5-10 dakika)
   - "Kurulum tamamlandı!" mesajını göreceksiniz

**İşlem:**
```
[1/3] Mevcut PyTorch kaldırılıyor...
[2/3] PyTorch GPU versiyonu kuruluyor...
[3/3] Test ediliyor...
```

---

### Yöntem 2: Manuel Kurulum (Terminal)

**PowerShell veya CMD açın:**

```powershell
# 1. Eski PyTorch'u kaldır
pip uninstall torch torchvision torchaudio -y

# 2. PyTorch GPU kur (CUDA 11.8 için)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Bekleme süresi:** ~5-10 dakika

---

## ✅ Kurulum Kontrolü

Kurulum bittikten sonra test edin:

### Test 1: Terminal'de
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Beklenen çıktı:**
```
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
```

### Test 2: Jupyter Notebook'ta

Browser'da açık Jupyter'da yeni hücre ekleyin:

```python
import torch

print("=" * 60)
print("GPU KONTROL")
print("=" * 60)
print(f"\nCUDA Mevcut: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("\n✅ SÜPER! GPU HAZIR!")
    print("   BERT eğitimi hızlı olacak: ~2-3 saat")
else:
    print("\n❌ Hala sorun var")
    print("   Bilgisayarı yeniden başlattınız mı?")

print("=" * 60)
```

**Shift + Enter** ile çalıştırın.

---

## 🐛 Sorun Giderme

### ❌ Hala "CUDA: False" Görüyorsanız:

#### Çözüm 1: Bilgisayarı Yeniden Başlatın
CUDA kurulumundan sonra mutlaka yeniden başlatmalısınız!

#### Çözüm 2: NVIDIA Driver Güncel mi?
```powershell
nvidia-smi
```
Bu komut GPU'nuzu göstermeli. Göstermiyorsa driver güncelleyin.

#### Çözüm 3: CUDA Versiyonu Uyumsuz
Eğer GPU'nuz çok eski veya çok yeni:
```powershell
# CUDA 12.x için:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎯 Adım Adım Checklist:

- [ ] CUDA Toolkit indirme bitti
- [ ] CUDA kurulumu yapıldı
- [ ] Bilgisayar yeniden başlatıldı ⚠️ ÖNEMLİ!
- [ ] `install_pytorch_gpu.bat` çalıştırıldı
- [ ] Test yapıldı: `torch.cuda.is_available()` → **True** görmeli
- [ ] GPU adı görünüyor

---

## ⚡ CUDA Kurulumundan Sonra:

1. **Bilgisayarı Yeniden Başlatın** (Mutlaka!)

2. **Bu Bat Dosyasını Çalıştırın:**
   - `install_pytorch_gpu.bat` (Proje klasöründe)
   - Çift tık → Bekle → Bitti!

3. **Test Edin:**
   - Jupyter'da yukarıdaki kodu çalıştırın
   - "CUDA: True" görmelisiniz

4. **BERT Eğitimine Başlayın:**
   ```bash
   python src/train_bert.py
   ```

---

## 📊 Beklenen Sonuç:

### Başarılı Kurulum:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3060 (veya sizinki)
GPU Bellek: 12.0 GB

✅ SÜPER! GPU HAZIR!
   BERT eğitimi hızlı olacak: ~2-3 saat
```

### Başarısız:
```
CUDA: False
❌ Hala sorun var
```

Bu durumda:
- Bilgisayarı yeniden başlattınız mı? (En önemli!)
- `nvidia-smi` çalışıyor mu?
- CUDA 11.8 mi kurdunuz?

---

## 🆘 Çok Karmaşık mı?

**Alternatif:** Google Colab kullanın!
- Kurulum yok
- Hemen başlayın
- Ücretsiz GPU

Söyleyin, Google Colab'ı göstereyim!

---

## ✨ Özet:

**ŞİMDİ:**
1. CUDA indirmesini bekleyin (~10-15 dk)
2. CUDA'yı kurun (~10 dk)
3. **Bilgisayarı yeniden başlatın** ⚠️
4. `install_pytorch_gpu.bat` çalıştırın
5. Test edin
6. BERT'i eğitin! 🎉

**TOPLAM SÜRE:** ~30-40 dakika

---

**CUDA kurulumu bitince bana haber verin, PyTorch kurulumuna geçeceğiz!** 😊

