"""
Generate Professional Visualizations for Presentation
=====================================================

This script creates all necessary charts and figures for your
academic presentation and report.

Usage:
    python src/12_generate_visuals.py

Expected time: 5-10 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("GENERATING PRESENTATION VISUALS")
print("=" * 70)

# Create reports directory if it doesn't exist
Path("../reports").mkdir(exist_ok=True)

# ============================================================================
# VISUAL 1: Model Comparison Bar Chart
# ============================================================================
print("\n[1/6] Creating model comparison chart...")

models = ['Baseline\n(TF-IDF)', 'LSTM\n(Word2Vec)', 'Ensemble\n(Combined)', 'Custom\nAttention']
accuracies = [86.04, 87.00, 88.40, 88.50]  # Update with your actual results
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylim([84, 90])
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.axhline(y=85, color='r', linestyle='--', alpha=0.5, label='Minimum Target: 85%')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('../reports/01_model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: reports/01_model_comparison.png")
plt.close()

# ============================================================================
# VISUAL 2: Ablation Study Results
# ============================================================================
print("\n[2/6] Creating ablation study chart...")

experiments = [
    'TF-IDF\nonly',
    '+ Deep\nLearning',
    '+ Domain\nFeatures',
    '+ Ensemble',
    '+ Custom\nAttention'
]
accuracies_ablation = [86.04, 87.00, 87.50, 88.40, 88.50]
improvements = [0, 0.96, 1.46, 2.36, 2.46]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Accuracy progression
bars1 = ax1.bar(experiments, accuracies_ablation, color='steelblue', alpha=0.8,
                edgecolor='black', linewidth=1.5)
for bar, acc in zip(bars1, accuracies_ablation):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylim([85, 90])
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Ablation Study: Cumulative Accuracy', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Right: Improvement bars
colors_imp = ['gray' if i == 0 else 'green' for i in improvements]
bars2 = ax2.bar(experiments, improvements, color=colors_imp, alpha=0.8,
                edgecolor='black', linewidth=1.5)
for bar, imp in zip(bars2, improvements):
    if imp > 0:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'+{imp:.2f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
ax2.set_title('Component Contributions', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../reports/02_ablation_study.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: reports/02_ablation_study.png")
plt.close()

# ============================================================================
# VISUAL 3: Per-Class Performance
# ============================================================================
print("\n[3/6] Creating per-class performance chart...")

classes = ['Access', 'Admin\nRights', 'HR\nSupport', 'Hardware', 
           'Internal\nProject', 'Misc.', 'Purchase', 'Storage']
f1_scores = [0.902, 0.831, 0.889, 0.848, 0.856, 0.831, 0.913, 0.914]

colors_class = ['green' if f1 > 0.88 else 'orange' if f1 > 0.84 else 'red' 
                for f1 in f1_scores]

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(classes, f1_scores, color=colors_class, alpha=0.8,
               edgecolor='black', linewidth=1.5)

# Add value labels
for bar, f1 in zip(bars, f1_scores):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{f1:.3f}',
            ha='left', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlim([0.80, 0.95])
ax.set_xlabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class Performance (F1-Score)', fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0.85, color='r', linestyle='--', alpha=0.5, label='Target: 0.85')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('../reports/03_per_class_performance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: reports/03_per_class_performance.png")
plt.close()

# ============================================================================
# VISUAL 4: Architecture Diagram (Text-based)
# ============================================================================
print("\n[4/6] Creating architecture overview...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Create architecture text
architecture_text = """
┌────────────────────────────────────────────────────────────┐
│                      INPUT TEXT                            │
│                  (IT Support Ticket)                       │
└─────────────────┬──────────────────────────────────────────┘
                  │
        ┌─────────┴────────┐
        │                  │
┌───────▼────────┐  ┌──────▼──────────┐
│  LSTM BRANCH   │  │ FEATURE BRANCH  │
│  (Semantic)    │  │ (Domain Expert) │
├────────────────┤  ├─────────────────┤
│ • Tokenization │  │ • Access Score  │
│ • Word2Vec     │  │ • Hardware Score│
│ • BiLSTM       │  │ • Urgency Score │
│ • ATTENTION ⭐  │  │ • 20+ Features⭐ │
│ • Dense(512)   │  │ • Dense(128)    │
└────────┬───────┘  └────────┬────────┘
         │                   │
         └────────┬──────────┘
                  │
         ┌────────▼────────┐
         │  FUSION LAYER   │
         │  (Concatenate)  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Dense(256)     │
         │  + Dropout      │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ OUTPUT (8 cls)  │
         │   Softmax       │
         └─────────────────┘

⭐ = ORIGINAL CONTRIBUTIONS
"""

ax.text(0.5, 0.5, architecture_text, 
        ha='center', va='center', 
        fontsize=11, 
        family='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

ax.set_title('Custom Hybrid Architecture', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../reports/04_architecture_diagram.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: reports/04_architecture_diagram.png")
plt.close()

# ============================================================================
# VISUAL 5: Training History (if available)
# ============================================================================
print("\n[5/6] Creating training history...")

try:
    with open("../models/custom_attention_history.pkl", "rb") as f:
        history = pickle.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy
    ax1.plot(history['accuracy'], label='Train', linewidth=2, marker='o')
    ax1.plot(history['val_accuracy'], label='Validation', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Model Accuracy Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history['loss'], label='Train', linewidth=2, marker='o')
    ax2.plot(history['val_loss'], label='Validation', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Model Loss Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/05_training_history.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: reports/05_training_history.png")
    plt.close()
except:
    print("  ⚠ Training history not found (train model first)")

# ============================================================================
# VISUAL 6: Feature Importance (Custom Features)
# ============================================================================
print("\n[6/6] Creating feature importance chart...")

feature_names = [
    'Access Score', 'Hardware Score', 'Software Score', 
    'Network Score', 'HR Score', 'Purchase Score', 'Storage Score',
    'Urgency Score', 'System Mentions', 'Text Length'
]
# Mock importance values (replace with actual if you compute them)
importance = [0.15, 0.14, 0.12, 0.10, 0.11, 0.09, 0.08, 0.13, 0.05, 0.03]

# Sort by importance
sorted_idx = np.argsort(importance)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = [importance[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(12, 8))
colors_imp = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
bars = ax.barh(sorted_features, sorted_importance, color=colors_imp, 
               alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, imp in zip(bars, sorted_importance):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{imp:.3f}',
            ha='left', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
ax.set_title('Custom Domain Features - Relative Importance', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../reports/06_feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: reports/06_feature_importance.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL VISUALS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files in reports/:")
print("  01_model_comparison.png")
print("  02_ablation_study.png")
print("  03_per_class_performance.png")
print("  04_architecture_diagram.png")
print("  05_training_history.png (if model trained)")
print("  06_feature_importance.png")
print("\n💡 Use these in your presentation and report!")
print("=" * 70)

