# 🖥️ GPU Kontrol Rehberi

## Hızlı Kontrol

Jupyter notebook'unuzda (zaten açık olan) **yeni bir hücre** ekleyip şunu çalıştırın:

```python
import torch

print("=" * 60)
print("GPU DURUM KONTROLÜ")
print("=" * 60)

cuda_available = torch.cuda.is_available()
print(f"\n✓ CUDA Mevcut: {cuda_available}")

if cuda_available:
    print(f"✓ CUDA Versiyonu: {torch.version.cuda}")
    print(f"✓ GPU Sayısı: {torch.cuda.device_count()}")
    print(f"✓ GPU Adı: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n🎉 HARIKA! GPU kullanıma hazır!")
    print("   BERT eğitimi hızlı olacak: ~2-3 saat")
else:
    print("\n⚠️  GPU bulunamadı - CPU versiyonu kurulu")
    print("   Eğitim yavaş olacak: ~8-12 saat")
    
print("=" * 60)
```

---

## Sonuçlara Göre:

### ✅ Eğer "CUDA Mevcut: True" diyorsa:
**SÜPER!** GPU hazır! Hiçbir şey yapmadan devam edebilirsiniz:

```bash
python src/train_bert.py
```

Eğitim ~2-3 saat sürecek.

---

### ❌ Eğer "CUDA Mevcut: False" diyorsa:

Bilgisayarınızda GPU olmasına rağmen Python onu göremiyordur. Bunun 2 nedeni var:

#### Neden 1: CUDA Kurulu Değil
NVIDIA CUDA Toolkit kurulmalı:
1. https://developer.nvidia.com/cuda-downloads adresine gidin
2. CUDA 11.8 veya 12.x sürümünü indirin
3. Kurun
4. Bilgisayarı yeniden başlatın

#### Neden 2: PyTorch CPU Versiyonu Kurulu
Şu anda CPU versiyonu kurulu, GPU versiyonunu kurmalısınız:

```bash
# Mevcut PyTorch'u kaldır
pip uninstall torch torchvision torchaudio

# GPU versiyonunu kur (CUDA 11.8 için)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

VEYA (CUDA 12.x için):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Sonra tekrar kontrol edin!

---

## Hangi GPU'nuz Var?

GPU modelinizi öğrenmek için:

**Windows'ta:**
1. Başlat → Görev Yöneticisi (Task Manager)
2. Performans sekmesi
3. GPU'ya tıklayın
4. Üstte GPU modelini göreceksiniz (örn: "NVIDIA GeForce RTX 3060")

**Veya komut satırından:**
```bash
nvidia-smi
```

---

## Önerilen GPU'lar:

| GPU | VRAM | BERT Eğitimi | Önerilen Batch Size |
|-----|------|--------------|---------------------|
| GTX 1650/1660 | 4-6 GB | ✅ Çalışır | 8 |
| RTX 2060/3050 | 6-8 GB | ✅ İyi | 16 |
| RTX 3060/4060 | 8-12 GB | ✅ Harika | 16-32 |
| RTX 3080/4080+ | 10+ GB | ✅ Mükemmel | 32+ |

---

## GPU Yoksa veya Çok Yavaşsa:

### Seçenek 1: Google Colab (ÜCRETSİZ GPU!)
1. https://colab.research.google.com
2. `src/03_bert_transformer.ipynb` yükle
3. Runtime → Change runtime → GPU
4. Runtime → Run all
5. ~2-3 saat bekle

### Seçenek 2: Kaggle (ÜCRETSİZ GPU!)
1. https://kaggle.com
2. New Notebook
3. Settings → Accelerator → GPU
4. Kodları yapıştır

### Seçenek 3: CPU ile Eğit (YAVAŞ ama mümkün)
```bash
python src/train_bert.py
```
~8-12 saat bekleyin ☕☕☕

---

## Hızlı Karar Ağacı:

```
GPU var mı? 
├─ EVET → CUDA kurulu mu?
│         ├─ EVET → PyTorch GPU versiyonu mu?
│         │         ├─ EVET → 🎉 HAZIRSINIZ!
│         │         └─ HAYIR → PyTorch GPU kur
│         └─ HAYIR → CUDA Toolkit kur
└─ HAYIR → Google Colab kullan
```

---

## Sonuç:

Jupyter notebook'ta yukarıdaki kodu çalıştırın ve sonuca göre:

- ✅ **True** → Devam edin!
- ❌ **False** → CUDA/PyTorch GPU kurulumu yapın veya Google Colab kullanın

**Not:** Bilgisayarınızda GPU olması yetmiyor, Python'un da onu görebilmesi gerek! 😊


