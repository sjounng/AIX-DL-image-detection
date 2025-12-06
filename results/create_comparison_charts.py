"""
Model Comparison Visualization Script

This script creates comprehensive comparison charts for all trained models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# Paths
metrics_dir = Path(__file__).parent / 'metrics'
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

# Model names and colors
models = ['simple_cnn', 'resnet50', 'efficientnet_b0', 'vgg16', 'convnext', 'ensemble_soft', 'ensemble_hard']
model_labels = ['SimpleCNN', 'ResNet50', 'EfficientNetB0', 'VGG16', 'ConvNeXt', 'Ensemble (Soft)', 'Ensemble (Hard)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#9B59B6', '#2ecc71', '#27ae60']

# Load all test results
results = []
for model in models:
    csv_path = metrics_dir / f'{model}_test_results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        results.append(df.iloc[0])
    else:
        print(f"Warning: {csv_path} not found")

results_df = pd.DataFrame(results)
results_df['model_name'] = model_labels

print("Loaded results:")
print(results_df)

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 12))

# 1. Accuracy Comparison (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(model_labels, results_df['test_accuracy'] * 100, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim(95, 100)
ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, results_df['test_accuracy'] * 100)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    if model_labels[i] == 'Ensemble (Soft)':  # Highlight best model
        ax1.text(bar.get_x() + bar.get_width()/2., height - 0.5,
                 '⭐', ha='center', va='top', fontsize=20)

# 2. Precision, Recall, F1 Comparison (Grouped Bar Chart)
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(model_labels))
width = 0.25

precision_bars = ax2.bar(x - width, results_df['test_precision'] * 100, width,
                          label='Precision', color='#FF6B6B', alpha=0.8, edgecolor='black')
recall_bars = ax2.bar(x, results_df['test_recall'] * 100, width,
                       label='Recall', color='#4ECDC4', alpha=0.8, edgecolor='black')
f1_bars = ax2.bar(x + width, results_df['test_f1'] * 100, width,
                   label='F1-Score', color='#45B7D1', alpha=0.8, edgecolor='black')

ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(model_labels, rotation=15, ha='right')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_ylim(95, 100)
ax2.grid(axis='y', alpha=0.3)

# 3. ROC AUC Comparison (Bar Chart)
ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(model_labels, results_df['roc_auc'], color=colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('ROC AUC', fontsize=12, fontweight='bold')
ax3.set_title('ROC AUC Comparison', fontsize=14, fontweight='bold', pad=20)
ax3.set_ylim(0.995, 1.001)
ax3.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, results_df['roc_auc'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Test Loss Comparison (Bar Chart)
ax4 = plt.subplot(2, 3, 4)
bars = ax4.bar(model_labels, results_df['test_loss'], color=colors, alpha=0.8, edgecolor='black')
ax4.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax4.set_title('Test Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
ax4.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, results_df['test_loss']):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 5. Radar Chart (All metrics normalized)
ax5 = plt.subplot(2, 3, 5, projection='polar')

# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot for each model
for i, (idx, row) in enumerate(results_df.iterrows()):
    values = [
        row['test_accuracy'],
        row['test_precision'],
        row['test_recall'],
        row['test_f1'],
        row['roc_auc']
    ]
    values += values[:1]  # Complete the circle

    ax5.plot(angles, values, 'o-', linewidth=2, label=model_labels[i], color=colors[i])
    ax5.fill(angles, values, alpha=0.15, color=colors[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(metrics, fontsize=10)
ax5.set_ylim(0.95, 1.0)
ax5.set_title('Overall Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7, ncol=1)
ax5.grid(True)

# 6. Model Efficiency (Accuracy vs Parameters)
ax6 = plt.subplot(2, 3, 6)

# Model parameters (approximate)
params = [2, 23, 4, 134, 28, 27, 27]  # in millions: SimpleCNN, ResNet50, EfficientNetB0, VGG16, ConvNeXt, Ensemble(Soft), Ensemble(Hard)
param_sizes = [p * 5 for p in params]  # Scale for bubble size

scatter = ax6.scatter(params, results_df['test_accuracy'] * 100,
                      s=param_sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=2)

# Add labels
for i, (x, y, label) in enumerate(zip(params, results_df['test_accuracy'] * 100, model_labels)):
    ax6.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                 fontsize=10, fontweight='bold')

ax6.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax6.set_title('Model Efficiency: Accuracy vs Size', fontsize=14, fontweight='bold', pad=20)
ax6.grid(alpha=0.3)
ax6.set_xscale('log')
ax6.set_ylim(96.5, 99.5)

# Add annotation for best efficiency (EfficientNetB0)
best_idx = model_labels.index('EfficientNetB0')
ax6.annotate('Best Efficiency ⭐',
             xy=(params[best_idx], results_df['test_accuracy'].iloc[best_idx] * 100),
             xytext=(20, -20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2),
             fontsize=10, fontweight='bold')

plt.suptitle('AI Image Detection: Comprehensive Model Comparison',
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = figures_dir / 'model_comparison_comprehensive.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comprehensive comparison saved: {output_path}")
plt.close()

# Create summary table figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Test Loss'])

# Sort by F1 score (primary metric)
sorted_results = results_df.sort_values('test_f1', ascending=False)
ranks = ['🥇', '🥈', '🥉', '4', '5', '6', '7']

for rank, (idx, row) in zip(ranks, sorted_results.iterrows()):
    table_data.append([
        rank,
        row['model_name'],
        f"{row['test_accuracy']*100:.2f}%",
        f"{row['test_precision']*100:.2f}%",
        f"{row['test_recall']*100:.2f}%",
        f"{row['test_f1']*100:.2f}%",
        f"{row['roc_auc']:.4f}",
        f"{row['test_loss']:.4f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.08, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(8):
    cell = table[(0, i)]
    cell.set_facecolor('#4ECDC4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 8):  # 7 models + header
    for j in range(8):
        cell = table[(i, j)]
        if i == 1:  # Best model
            cell.set_facecolor('#E8F8F5')
        elif i % 2 == 0:
            cell.set_facecolor('#F8F9FA')

plt.title('AI Image Detection Model Performance Summary',
          fontsize=16, fontweight='bold', pad=20)

output_path = figures_dir / 'model_comparison_table.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Summary table saved: {output_path}")
plt.close()

print("\n" + "="*60)
print("Model Comparison Visualizations Created!")
print("="*60)
print(f"1. Comprehensive comparison: {figures_dir / 'model_comparison_comprehensive.png'}")
print(f"2. Summary table: {figures_dir / 'model_comparison_table.png'}")
print("="*60)
