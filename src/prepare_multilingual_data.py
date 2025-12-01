"""
Türkçe örnekleri ekleyerek çok dilli veri seti oluştur
Bu script, orijinal İngilizce veri + Türkçe örnekleri birleştirir
"""
import pandas as pd

print("=" * 60)
print("ÇOK DİLLİ VERİ SETİ HAZIRLAMA")
print("=" * 60)

# 1. Orijinal veriyi yükle
print("\n[1/4] Orijinal İngilizce veri yükleniyor...")
# Try different file names with proper path handling
import os
from pathlib import Path

# Get script directory and build absolute paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

csv_path = data_dir / "cleaned_data.csv"
xlsx_path = data_dir / "all_tickets_processed_improved_v3.xlsx"

if csv_path.exists():
    df_original = pd.read_csv(csv_path)
    print(f"   ✓ CSV dosyası yüklendi: {csv_path}")
elif xlsx_path.exists():
    df_original = pd.read_excel(xlsx_path)
    print(f"   ✓ Excel dosyası yüklendi: {xlsx_path}")
    # Clean column names if needed
    if 'text' not in df_original.columns or 'label' not in df_original.columns:
        print("   ⚠️  Sütun isimleri kontrol ediliyor...")
        print(f"   Mevcut sütunlar: {df_original.columns.tolist()}")
        # Try to find text and label columns
        text_col = [col for col in df_original.columns if 'text' in col.lower() or 'description' in col.lower()]
        label_col = [col for col in df_original.columns if 'label' in col.lower() or 'category' in col.lower()]
        if text_col and label_col:
            df_original = df_original.rename(columns={text_col[0]: 'text', label_col[0]: 'label'})
        else:
            raise ValueError(f"Text ve Label sütunları bulunamadı. Sütunlar: {df_original.columns.tolist()}")
else:
    raise FileNotFoundError(f"Veri dosyası bulunamadı!\nAradığım: {csv_path}\nveya: {xlsx_path}")
print(f"   İngilizce örnekler: {len(df_original):,}")
print(f"   Kategoriler: {df_original['label'].unique().tolist()}")

# 2. Türkçe örnekleri yükle
print("\n[2/4] Türkçe örnekler yükleniyor...")
turkish_path = data_dir / "turkish_tickets_extended.csv"
df_turkish = pd.read_csv(turkish_path)
print(f"   ✓ Türkçe örnekler: {len(df_turkish):,}")

# 3. Birleştir
print("\n[3/4] Veri setleri birleştiriliyor...")
df_combined = pd.concat([df_original, df_turkish], ignore_index=True)
print(f"   Toplam örnekler: {len(df_combined):,}")

# 4. Kaydet
output_path = data_dir / "cleaned_data_multilingual.csv"
df_combined.to_csv(output_path, index=False)
print(f"\n[4/4] ✓ Kaydedildi: {output_path}")

# İstatistikler
print("\n" + "=" * 60)
print("SONUÇ İSTATİSTİKLERİ")
print("=" * 60)
print(f"\nToplam veri: {len(df_combined):,} ticket")
print(f"İngilizce: {len(df_original):,} ({len(df_original)/len(df_combined)*100:.1f}%)")
print(f"Türkçe: {len(df_turkish):,} ({len(df_turkish)/len(df_combined)*100:.1f}%)")

print("\nKategori dağılımı:")
print(df_combined['label'].value_counts())

print("\n✅ Veri hazırlama tamamlandı!")
print("   Şimdi BERT modelini bu veri ile eğitebilirsiniz.")
print("\n📝 Sonraki adım:")
print("   1. Jupyter Notebook aç: src/03_bert_transformer.ipynb")
print("   2. Veri yükleme kısmını değiştir:")
print("      df = pd.read_csv('../data/cleaned_data_multilingual.csv')")
print("   3. Tüm cell'leri çalıştır")

