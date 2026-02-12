import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Define colors - All white for IEEE conference paper standards
input_color = 'white'
cnn_color = 'white'
embedding_color = 'white'
transformer_color = 'white'
attention_color = 'white'
ffn_color = 'white'
pooling_color = 'white'
fc_color = 'white'
output_color = 'white'

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, subtext=None, color='white', edge_color='black'):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", edgecolor=edge_color,
                         facecolor=color, linewidth=2.5, zorder=2)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.15, text, ha='center', va='center', 
                fontsize=20, fontweight='bold', zorder=3)
        ax.text(x, y - 0.2, subtext, ha='center', va='center', 
                fontsize=16, zorder=3, style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=20, fontweight='bold', zorder=3)

# Helper function to create arrows
def create_arrow(ax, x, y_start, y_end):
    arrow = FancyArrowPatch((x, y_start), (x, y_end),
                           arrowstyle='->', mutation_scale=25,
                           color='black', linewidth=2.5, zorder=1)
    ax.add_patch(arrow)

# 1. Input - Sequence of Window Feature Vectors
# Input box with text inside
input_box = FancyBboxPatch((1.2, 13.5), 7.6, 1.2,
                          boxstyle="round,pad=0.1", edgecolor='black',
                          facecolor=input_color, linewidth=2.5, zorder=2)
ax.add_patch(input_box)
ax.text(5, 14.25, "Sequence of Window Feature Vectors", 
        ha='center', va='center', fontsize=18, fontweight='bold', zorder=4)
ax.text(5, 13.8, "(Sequence Length × 18)", 
        ha='center', va='center', fontsize=15, zorder=4)

create_arrow(ax, 5, 13.45, 13.0)

# 2. 1D Convolution Layers
create_box(ax, 5, 12.5, 7, 1, "1D Convolution Layers", 
          "(Local burst pattern learning)", cnn_color, 'black')
create_arrow(ax, 5, 11.95, 11.3)

# 3. Feature Embedding Layer
create_box(ax, 5, 10.8, 7, 0.8, "Feature Embedding Layer", None, embedding_color, 'black')
create_arrow(ax, 5, 10.35, 9.3)

# 4. Transformer Encoder Blocks
# Outer container
transformer_container = FancyBboxPatch((1.5, 6.8), 7, 2.4,
                                      boxstyle="round,pad=0.15", edgecolor='black',
                                      facecolor=transformer_color, linewidth=2.5, zorder=1)
ax.add_patch(transformer_container)
ax.text(5, 9.0, "Transformer Encoder Blocks", 
        ha='center', va='center', fontsize=20, fontweight='bold', zorder=3)
ax.text(5, 8.65, "(Self-Attention + FFN)", 
        ha='center', va='center', fontsize=16, style='italic', zorder=3)

# Multi-Head Self-Attention
attention_box = FancyBboxPatch((2.0, 7.75), 6, 0.6,
                              boxstyle="round,pad=0.08", edgecolor='black',
                              facecolor=attention_color, linewidth=2, zorder=3)
ax.add_patch(attention_box)
ax.text(5, 8.05, "Multi-Head Self-Attention", 
        ha='center', va='center', fontsize=18, fontweight='bold', color='black', zorder=4)

# Feed Forward Network
ffn_box = FancyBboxPatch((2.0, 7.0), 6, 0.6,
                        boxstyle="round,pad=0.08", edgecolor='black',
                        facecolor=ffn_color, linewidth=2, zorder=3)
ax.add_patch(ffn_box)
ax.text(5, 7.3, "Feed Forward Network", 
        ha='center', va='center', fontsize=18, fontweight='bold', color='black', zorder=4)

create_arrow(ax, 5, 6.7, 6.1)

# 5. Global Average Pooling
create_box(ax, 5, 5.6, 7, 0.8, "Global Average Pooling", None, pooling_color, 'black')
create_arrow(ax, 5, 5.15, 4.5)

# 6. Fully Connected Layer
create_box(ax, 5, 4.0, 7, 0.8, "Fully Connected Layer", None, fc_color, 'black')
create_arrow(ax, 5, 3.55, 2.65)

# 7. Risk Score Output - simple text only
# Output box
output_box = FancyBboxPatch((1.5, 1.4), 7, 1.1,
                           boxstyle="round,pad=0.1", edgecolor='black',
                           facecolor=output_color, linewidth=2.5, zorder=2)
ax.add_patch(output_box)
ax.text(5, 2.1, "Risk Score Output", 
        ha='center', va='center', fontsize=20, fontweight='bold', zorder=4)
ax.text(5, 1.7, "(0 = Normal → 2 = Failure severity scale)", 
        ha='center', va='center', fontsize=14, zorder=4)

# Set white background
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig('architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Architecture diagram saved as architecture.png")
