import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Define colors
box_color = '#2c5f8d'
arrow_color = '#2c5f8d'
bg_color = 'white'

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, subtext=None):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", 
                          edgecolor=box_color, facecolor=bg_color,
                          linewidth=3, zorder=2)
    ax.add_patch(box)
    ax.text(x, y + 0.1, text, ha='center', va='center', 
            fontsize=19, fontweight='bold', zorder=3)
    if subtext:
        ax.text(x, y - 0.25, subtext, ha='center', va='center', 
                fontsize=12, zorder=3)

# Helper function to create arrows
def create_arrow(ax, x, y_start, y_end):
    arrow = FancyArrowPatch((x, y_start), (x, y_end),
                           arrowstyle='->', mutation_scale=30,
                           color=arrow_color, linewidth=3, zorder=1)
    ax.add_patch(arrow)

# Step 1: BGL Raw Logs
create_box(ax, 5, 18.5, 3.5, 1, "BGL Raw Logs")
create_arrow(ax, 5, 18.0, 16.5)

# Step 2: Log Parsing & Structuring
create_box(ax, 5, 16, 4, 1, "Log Parsing & Structuring")
create_arrow(ax, 5, 15.5, 13.9)

# Step 3: 5-Min Statistical Windowing
create_box(ax, 5, 13.3, 4, 1.2, "5-Min Statistical Windowing\n(18 Features Extracted)")
create_arrow(ax, 5, 12.7, 11.7)

# Step 4: Episode Construction
create_box(ax, 5, 11.2, 3.5, 1, "Episode Construction")
create_arrow(ax, 5, 10.7, 9.9)

# Step 5: Hierarchical Labeling (with label container)
# Labels container box (outer box)
label_container = FancyBboxPatch((1.3, 8.4), 7.4, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor=box_color, facecolor='white',
                                linewidth=3, zorder=2)
ax.add_patch(label_container)

# Add title for labeling section
ax.text(5, 9.6, "Hierarchical Labeling", ha='center', va='center',
        fontsize=19, fontweight='bold', color='black', zorder=4)

# Labels: Normal, Pre-Failure, Failure (no background color, just text with borders)
# Normal
normal_box = FancyBboxPatch((1.8, 8.65), 1.8, 0.6,
                           boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor=box_color,
                           linewidth=2, zorder=3)
ax.add_patch(normal_box)
ax.text(2.7, 8.95, "Normal", ha='center', va='center',
        fontsize=15, fontweight='bold', color='black', zorder=4)

# Pre-Failure
pre_failure_box = FancyBboxPatch((4.1, 8.65), 1.8, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=box_color,
                                linewidth=2, zorder=3)
ax.add_patch(pre_failure_box)
ax.text(5.0, 8.95, "Pre-Failure", ha='center', va='center',
        fontsize=15, fontweight='bold', color='black', zorder=4)

# Failure
failure_box = FancyBboxPatch((6.4, 8.65), 1.8, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor=box_color,
                            linewidth=2, zorder=3)
ax.add_patch(failure_box)
ax.text(7.3, 8.95, "Failure", ha='center', va='center',
        fontsize=15, fontweight='bold', color='black', zorder=4)

# Arrow from label container to Sequence Construction
create_arrow(ax, 5, 8.4, 7.7)

# Step 6: Sequence Construction (increased box size)
create_box(ax, 5, 7.0, 5, 1.4, "Sequence Construction\n(Non-overlapping window sequences)")
create_arrow(ax, 5, 6.3, 5.4)

# Step 7: Hybrid CNN-Transformer Model
create_box(ax, 5, 4.8, 5.5, 1, "Train the Model")
create_arrow(ax, 5, 4.3, 3.3)

# Step 8: Failure Risk Prediction
create_box(ax, 5, 2.7, 4.5, 1, "Failure Risk Prediction")

# Set white background
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('flowchart_final.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Final flowchart saved as flowchart_final.png")