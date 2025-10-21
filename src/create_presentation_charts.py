"""
Hocaya Sunum için Grafikler Oluştur
Word'e yapıştırılacak görsel grafikler
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("SUNUM GRAFİKLERİ OLUŞTURULUYOR")
print("=" * 70)

# 1. MODEL PERFORMANS KARŞILAŞTIRMA GRAFİĞİ
print("\n[1/5] Model performans grafiği...")
fig, ax = plt.subplots(figsize=(12, 7))

models = ['Baseline\n(TF-IDF+LogReg)', 'LSTM\n(Word2Vec+BiLSTM)', 
          'Ensemble\n(Base+LSTM)', 'BERT\n(Fine-tuned)']
accuracies = [86.04, 87.00, 88.40, 88.82]
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Değerleri bar üstüne yaz
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performans Karşılaştırması', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([84, 91])
ax.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Hedef: 85%')
ax.axhline(y=88, color='green', linestyle='--', alpha=0.5, label='Başarı: 88%')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/01_model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ reports/01_model_comparison.png")
plt.close()


# 2. İYİLEŞTİRME GRAFİĞİ
print("[2/5] İyileştirme progress grafiği...")
fig, ax = plt.subplots(figsize=(12, 7))

steps = ['Baseline', 'LSTM', 'Ensemble', 'BERT']
acc_values = [86.04, 87.00, 88.40, 88.82]

ax.plot(steps, acc_values, marker='o', linewidth=3, markersize=12, color='#2ecc71')
ax.fill_between(range(len(steps)), acc_values, alpha=0.3, color='#2ecc71')

for i, (step, acc) in enumerate(zip(steps, acc_values)):
    ax.text(i, acc + 0.4, f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model İyileştirme Süreci', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([85, 90])
ax.grid(True, alpha=0.3)
ax.axhline(y=88, color='orange', linestyle='--', alpha=0.5, label='İyileştirme hedefi')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('reports/02_improvement_progress.png', dpi=300, bbox_inches='tight')
print("   ✓ reports/02_improvement_progress.png")
plt.close()


# 3. SINIF BAZINDA PERFORMANS (BERT)
print("[3/5] Sınıf bazında performans grafiği...")
fig, ax = plt.subplots(figsize=(14, 8))

classes = ['Purchase', 'Storage', 'Access', 'HR Support', 
           'Internal Project', 'Hardware', 'Miscellaneous', 'Admin Rights']
f1_scores = [0.938, 0.922, 0.915, 0.900, 0.887, 0.877, 0.847, 0.840]

colors_gradient = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(classes)))
bars = ax.barh(classes, f1_scores, color=colors_gradient, edgecolor='black', linewidth=1.2)

for bar, score in zip(bars, f1_scores):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{score:.3f}',
            ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_title('Sınıf Bazında Performans (BERT Model)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim([0.8, 1.0])
ax.axvline(x=0.90, color='green', linestyle='--', alpha=0.5, label='Hedef: 0.90')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('reports/03_class_performance_bert.png', dpi=300, bbox_inches='tight')
print("   ✓ reports/03_class_performance_bert.png")
plt.close()


# 4. VERİ DAĞILIMI GRAFİĞİ
print("[4/5] Veri dağılımı grafiği...")
fig, ax = plt.subplots(figsize=(12, 8))

categories = ['Hardware', 'HR Support', 'Access', 'Miscellaneous', 
              'Storage', 'Purchase', 'Internal Project', 'Admin Rights']
counts = [13617, 10915, 7125, 7060, 2777, 2464, 2119, 1760]

colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%',
                                    colors=colors, startangle=90,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})

ax.set_title('Veri Seti Dağılımı (47,837 Ticket)', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('reports/04_data_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ reports/04_data_distribution.png")
plt.close()


# 5. MODEL KARŞILAŞTIRMA - DETAYLI
print("[5/5] Detaylı model karşılaştırma...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Sol: Accuracy, Precision, Recall, F1
models_short = ['Baseline', 'LSTM', 'Ensemble', 'BERT']
metrics = {
    'Accuracy': [86.04, 87.00, 88.40, 88.82],
    'Precision': [87, 87, 89, 89],
    'Recall': [86, 87, 88, 89],
    'F1-Score': [86, 87, 88, 89]
}

x = np.arange(len(models_short))
width = 0.2

for i, (metric, values) in enumerate(metrics.items()):
    ax1.bar(x + i*width, values, width, label=metric, alpha=0.8)

ax1.set_xlabel('Modeller', fontsize=12, fontweight='bold')
ax1.set_ylabel('Skor (%)', fontsize=12, fontweight='bold')
ax1.set_title('Tüm Metriklerde Karşılaştırma', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(models_short)
ax1.legend(fontsize=10)
ax1.set_ylim([84, 92])
ax1.grid(True, alpha=0.3, axis='y')

# Sağ: Eğitim süresi
training_times = [5/60, 30/60, 0, 150/60]  # Saat cinsinden
colors_time = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

bars2 = ax2.bar(models_short, training_times, color=colors_time, alpha=0.8, edgecolor='black')
for bar, time in zip(bars2, training_times):
    height = bar.get_height()
    if time > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{time:.1f}h' if time >= 1 else f'{int(time*60)}m',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., 0.05,
                'Instant', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Eğitim Süresi (Saat)', fontsize=12, fontweight='bold')
ax2.set_title('Eğitim Süreleri Karşılaştırması', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/05_detailed_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ reports/05_detailed_comparison.png")
plt.close()


print("\n" + "=" * 70)
print("TÜM GRAFİKLER OLUŞTURULDU!")
print("=" * 70)
print("\nOluşturulan dosyalar:")
print("  1. reports/01_model_comparison.png - Model performansları")
print("  2. reports/02_improvement_progress.png - İyileştirme süreci")
print("  3. reports/03_class_performance_bert.png - Sınıf bazında performans")
print("  4. reports/04_data_distribution.png - Veri dağılımı")
print("  5. reports/05_detailed_comparison.png - Detaylı karşılaştırma")
print("\nBu grafikleri Word dökümanınıza ekleyebilirsiniz!")
print("=" * 70)

