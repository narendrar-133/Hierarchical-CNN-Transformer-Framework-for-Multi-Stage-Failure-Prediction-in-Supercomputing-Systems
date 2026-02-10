import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ['CNN', 'LSTM', 'Transformer', 'Hybrid\nCNN-Transformer']
balanced_acc = [0.8928, 0.9532, 0.9474, 0.8956]
mae = [0.1577, 0.1345, 0.1466, 0.1095]
rmse = [0.3853, 0.2829, 0.3198, 0.3224]
within_range = [84.3, 88.7, 86.2, 90.28]

# Define colors - professional palette
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']

# Bar width and positions
x = np.arange(len(models))
bar_width = 0.6

# =====================================================================
# Figure 1: Balanced Accuracy
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(x, balanced_acc, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Balanced Accuracy', fontsize=16, fontweight='bold')
ax1.set_xlabel('Models', fontsize=16, fontweight='bold')
ax1.set_title('Balanced Accuracy Comparison', fontsize=18, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=14)
ax1.set_ylim([0.85, 0.97])
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
ax1.tick_params(axis='both', labelsize=13)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, balanced_acc)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('balanced_accuracy.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Figure 1: balanced_accuracy.png saved")
plt.close()

# =====================================================================
# Figure 2: MAE (Mean Absolute Error)
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars2 = ax2.bar(x, mae, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Models', fontsize=16, fontweight='bold')
ax2.set_title('MAE Comparison', fontsize=18, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=14)
ax2.set_ylim([0, 0.18])
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
ax2.tick_params(axis='both', labelsize=13)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, mae)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('mae_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Figure 2: mae_comparison.png saved")
plt.close()

# =====================================================================
# Figure 3: RMSE (Root Mean Square Error)
# =====================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars3 = ax3.bar(x, rmse, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Root Mean Square Error (RMSE)', fontsize=16, fontweight='bold')
ax3.set_xlabel('Models', fontsize=16, fontweight='bold')
ax3.set_title('RMSE Comparison', fontsize=18, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=14)
ax3.set_ylim([0, 0.42])
ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
ax3.tick_params(axis='both', labelsize=13)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, rmse)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.008,
            f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Figure 3: rmse_comparison.png saved")
plt.close()

# =====================================================================
# Figure 4: Within ±0.5 (%)
# =====================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))
bars4 = ax4.bar(x, within_range, bar_width, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Percentage Within ±0.5 (%)', fontsize=16, fontweight='bold')
ax4.set_xlabel('Models', fontsize=16, fontweight='bold')
ax4.set_title('Prediction Accuracy Within ±0.5 Range', fontsize=18, fontweight='bold', pad=20)
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=14)
ax4.set_ylim([80, 95])
ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
ax4.tick_params(axis='both', labelsize=13)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars4, within_range)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('within_range_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Figure 4: within_range_comparison.png saved")
plt.close()

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print("\nGenerated files:")
print("1. balanced_accuracy.png")
print("2. mae_comparison.png")
print("3. rmse_comparison.png")
print("4. within_range_comparison.png")
print("\nAll images are ready for use in your IEEE conference paper.")
