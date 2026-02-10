import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ['CNN', 'LSTM', 'Transformer', 'Hybrid\nCNN-Transformer']
balanced_acc = [0.8928, 0.9532, 0.9474, 0.8956]
mae = [0.1577, 0.1345, 0.1466, 0.1095]
rmse = [0.3853, 0.2829, 0.3198, 0.3224]
within_range = [84.3, 88.7, 86.2, 90.28]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparative Performance of Baseline and Proposed Models', 
             fontsize=18, fontweight='bold', y=0.995)

# Define colors - professional palette
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']

# Bar width and positions
x = np.arange(len(models))
bar_width = 0.6

# 1. Balanced Accuracy
ax1 = axes[0, 0]
bars1 = ax1.bar(x, balanced_acc, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylim([0.85, 0.96])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(axis='both', labelsize=11)
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, balanced_acc)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. MAE (Mean Absolute Error) - Lower is better
ax2 = axes[0, 1]
bars2 = ax2.bar(x, mae, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('MAE (Lower is Better)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=12)
ax2.set_ylim([0, 0.18])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='both', labelsize=11)
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, mae)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. RMSE (Root Mean Square Error) - Lower is better
ax3 = axes[1, 0]
bars3 = ax3.bar(x, rmse, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('RMSE (Lower is Better)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Models', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=12)
ax3.set_ylim([0, 0.42])
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.tick_params(axis='both', labelsize=11)
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, rmse)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Within ±0.5 (%)
ax4 = axes[1, 1]
bars4 = ax4.bar(x, within_range, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Within ±0.5 (%)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Models', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=12)
ax4.set_ylim([80, 95])
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.tick_params(axis='both', labelsize=11)
# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars4, within_range)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Comparison plot saved as model_comparison.png")
plt.show()
