"""
Tüm veri kaynaklarını birleştir ve final training set'i oluştur
1000+ örnek hedefine ulaşmak için
"""
import pandas as pd

print("=" * 70)
print("KAPSAMLI VERİ SETİ OLUŞTURMA - 1000+ ÖRNEK")
print("=" * 70)

# 1. Orijinal veri (İngilizce)
print("\n[1/3] Orijinal veri yükleniyor...")
df_original = pd.read_csv("../data/cleaned_data.csv")
print(f"   Original: {len(df_original):,} samples")

# 2. Yeni kapsamlı veri (Türkçe + İngilizce dengeli)
print("\n[2/3] Yeni kapsamlı veri yükleniyor...")
df_new1 = pd.read_csv("../data/comprehensive_training_data.csv")
if 'language' in df_new1.columns:
    df_new1 = df_new1[['text', 'label']]

df_new2 = pd.read_csv("../data/software_network_boost.csv")
df_new3 = pd.read_csv("../data/software_network_massive_boost.csv")
df_new4 = pd.read_csv("../data/software_network_final_boost.csv")

df_new = pd.concat([df_new1, df_new2, df_new3, df_new4], ignore_index=True)
print(f"   New comprehensive: {len(df_new):,} samples")

# Kategori dağılımı
print("\n   Kategori dağılımı (yeni veri):")
print(df_new['label'].value_counts())

# 3. Birleştir
print("\n[3/3] Veri setleri birleştiriliyor...")
df_combined = pd.concat([df_original, df_new], ignore_index=True)

# Duplikasyon kontrolü
initial_size = len(df_combined)
df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
removed = initial_size - len(df_combined)

print(f"   Toplam: {len(df_combined):,} samples")
print(f"   Duplikasyon temizlendi: {removed} satır")

# 4. Kaydet
output_path = "../data/cleaned_data_multilingual_v2.csv"
df_combined.to_csv(output_path, index=False)
print(f"\n✅ Kaydedildi: {output_path}")

# İstatistikler
print("\n" + "=" * 70)
print("FINAL VERİ SETİ İSTATİSTİKLERİ")
print("=" * 70)
print(f"\nToplam veri: {len(df_combined):,} ticket")
print(f"Orijinal: {len(df_original):,}")
print(f"Yeni eklenen: {len(df_new):,}")

print("\n📊 Kategori Dağılımı (Final):")
category_counts = df_combined['label'].value_counts()
print(category_counts)

print("\n📈 Kategori Bazında:")
for cat, count in category_counts.items():
    percentage = (count / len(df_combined)) * 100
    print(f"   {cat:25s}: {count:5d} (%{percentage:5.2f})")

print("\n✅ Veri hazırlama tamamlandı!")
print("\n📝 Sonraki adımlar:")
print("   1. BERT notebook'unda veri yolunu güncelle:")
print("      df = pd.read_csv('../data/cleaned_data_multilingual_v2.csv')")
print("   2. Modeli yeniden eğit")
print("   3. Test et ve karşılaştır")

