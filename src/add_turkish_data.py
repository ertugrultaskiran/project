"""
Türkçe örnekleri mevcut veri setine ekle ve modelleri yeniden eğit
"""
import pandas as pd

# Mevcut temizlenmiş veriyi yükle
df_original = pd.read_csv("../data/cleaned_data.csv")
print(f"Original dataset: {df_original.shape}")
print(f"Categories: {df_original['label'].unique()}")

# Türkçe örnekleri yükle
df_turkish = pd.read_csv("../data/turkish_samples.csv")
print(f"\nTurkish samples: {df_turkish.shape}")

# Birleştir
df_combined = pd.concat([df_original, df_turkish], ignore_index=True)
print(f"\nCombined dataset: {df_combined.shape}")

# Kaydet (orijinalin üzerine yazmayalım, yeni dosya oluşturalım)
df_combined.to_csv("../data/cleaned_data_with_turkish.csv", index=False)
print("✅ Saved to: data/cleaned_data_with_turkish.csv")

# Sınıf dağılımı
print("\nClass distribution:")
print(df_combined['label'].value_counts())

