"""
EPISODE-BASED WINDOW LABELING (OPTIMIZED)
Run this AFTER create_windows_CORRECTED.py

OPTIMIZATIONS:
- Vectorized operations instead of nested loops
- Batch processing
- Efficient indexing
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ===============================
# CONFIG
# ===============================
EPISODE_LENGTH = 8  # Number of windows per episode (8 windows = 40 min)
EPISODE_STRIDE = 8  # NO overlap for labeling

USE_3_CLASSES = True  # 3 classes: Normal, Pre-Failure, Failure

INPUT_WINDOWS_CSV = "windows_with_features.csv"
OUTPUT_LABELED_CSV = "final_labeled_windows.csv"
OUTPUT_METADATA = "labeling_metadata.pkl"

# Labeling thresholds
FAILURE_THRESHOLD = 20
PRE_FAILURE_THRESHOLD = 5

print("="*70)
print("EPISODE-BASED WINDOW LABELING (OPTIMIZED)")
print("="*70)
print(f"Episode length: {EPISODE_LENGTH} windows (~{EPISODE_LENGTH * 5} minutes)")
print(f"Episode stride: {EPISODE_STRIDE} windows (NO overlap)")
print(f"Class system: {'3 classes' if USE_3_CLASSES else '4 classes'}")

# ===============================
# LOAD WINDOWS
# ===============================
print("\n[1/3] Loading windows...")
start_time = datetime.now()

windows_df = pd.read_csv(INPUT_WINDOWS_CSV)
windows_df = windows_df.sort_values(['Node', 'Time']).reset_index(drop=True)

print(f"  Loaded {len(windows_df):,} windows")
print(f"  Nodes: {windows_df['Node'].nunique()}")
print(f"  Features per window: {len(windows_df.columns) - 2}")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  Time: {elapsed:.1f}s")

# ===============================
# OPTIMIZED EPISODE LABELING
# ===============================
print(f"\n[2/3] Labeling windows (VECTORIZED)...")
start_time = datetime.now()

# Pre-allocate label array
windows_df['Episode_Label'] = 0  # Default to Normal

episode_stats_list = []
nodes_processed = 0

# Get unique nodes
unique_nodes = windows_df['Node'].unique()
total_nodes = len(unique_nodes)

print(f"  Processing {total_nodes} nodes...")

for node in unique_nodes:
    # Get all windows for this node at once
    node_mask = windows_df['Node'] == node
    node_indices = np.where(node_mask)[0]
    
    if len(node_indices) < EPISODE_LENGTH:
        # Too few windows, already labeled as Normal (0)
        nodes_processed += 1
        continue
    
    # Extract node data once
    fatal_counts = windows_df.loc[node_indices, 'fatal_count'].values
    error_counts = windows_df.loc[node_indices, 'error_count'].values
    
    # Process episodes in batches
    num_windows = len(node_indices)
    
    for start_idx in range(0, num_windows - EPISODE_LENGTH + 1, EPISODE_STRIDE):
        end_idx = start_idx + EPISODE_LENGTH
        
        # Get episode data (vectorized)
        episode_fatal = fatal_counts[start_idx:end_idx]
        episode_error = error_counts[start_idx:end_idx]
        
        # FAST episode analysis (all vectorized)
        max_fatal = np.max(episode_fatal)
        max_error = np.max(episode_error)
        total_fatal = np.sum(episode_fatal)
        mean_fatal = np.mean(episode_fatal)
        
        # Trend (vectorized)
        mid = EPISODE_LENGTH // 2
        fatal_trend = np.mean(episode_fatal[mid:]) - np.mean(episode_fatal[:mid])
        
        # FAST labeling decision
        if USE_3_CLASSES:
            if max_fatal >= FAILURE_THRESHOLD:
                label = 2  # Failure
            elif (max_fatal >= PRE_FAILURE_THRESHOLD or 
                  total_fatal >= 15 or
                  (fatal_trend > 3 and mean_fatal > 2) or
                  max_error >= 20):
                label = 1  # Pre-Failure
            else:
                label = 0  # Normal
        else:
            if max_fatal >= 25:
                label = 3
            elif max_fatal >= 15:
                label = 2
            elif max_fatal >= 8 or fatal_trend > 5:
                label = 1
            else:
                label = 0
        
        # Assign label to ALL windows in episode (vectorized)
        episode_global_indices = node_indices[start_idx:end_idx]
        windows_df.loc[episode_global_indices, 'Episode_Label'] = label
        
        # Track stats (optional, for analysis)
        if len(episode_stats_list) < 10000:  # Limit memory
            episode_stats_list.append({
                'max_fatal': int(max_fatal),
                'fatal_trend': float(fatal_trend),
                'label': int(label)
            })
    
    nodes_processed += 1
    
    # Progress indicator
    if nodes_processed % 100 == 0 or nodes_processed == total_nodes:
        pct = 100 * nodes_processed / total_nodes
        print(f"    Processed {nodes_processed}/{total_nodes} nodes ({pct:.1f}%)")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  ✓ Labeled {len(windows_df):,} windows in {elapsed:.1f}s")

# ===============================
# ANALYZE LABELS
# ===============================
print("\n  Label distribution:")

if USE_3_CLASSES:
    label_names = {0: 'Normal', 1: 'Pre-Failure', 2: 'Failure'}
else:
    label_names = {0: 'Normal', 1: 'Early Warning', 2: 'Pre-Failure', 3: 'Failure'}

for label in sorted(windows_df['Episode_Label'].unique()):
    count = (windows_df['Episode_Label'] == label).sum()
    pct = 100 * count / len(windows_df)
    name = label_names.get(label, f'Label {label}')
    print(f"    {name}: {count:,} ({pct:.2f}%)")

# Quick stats
print("\n  Average features by label:")
for label in sorted(windows_df['Episode_Label'].unique()):
    label_data = windows_df[windows_df['Episode_Label'] == label]
    print(f"\n  {label_names[label]} (Label {label}):")
    print(f"    Count: {len(label_data):,}")
    print(f"    Mean fatal_count: {label_data['fatal_count'].mean():.2f}")
    print(f"    Max fatal_count: {label_data['fatal_count'].max()}")

# Imbalance check
label_counts = windows_df['Episode_Label'].value_counts()
imbalance_ratio = label_counts.max() / label_counts.min() if label_counts.min() > 0 else float('inf')
print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}:1")

# ===============================
# SAVE
# ===============================
print(f"\n[3/3] Saving...")
start_time = datetime.now()

windows_df.to_csv(OUTPUT_LABELED_CSV, index=False)

metadata = {
    'episode_length': EPISODE_LENGTH,
    'episode_stride': EPISODE_STRIDE,
    'num_windows': len(windows_df),
    'num_nodes': windows_df['Node'].nunique(),
    'use_3_classes': USE_3_CLASSES,
    'label_names': label_names,
    'label_distribution': {
        int(label): int(count) 
        for label, count in windows_df['Episode_Label'].value_counts().items()
    },
    'thresholds': {
        'failure': FAILURE_THRESHOLD,
        'pre_failure': PRE_FAILURE_THRESHOLD
    },
    'created_at': datetime.now().isoformat()
}

with open(OUTPUT_METADATA, 'wb') as f:
    pickle.dump(metadata, f)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  ✓ Saved in {elapsed:.1f}s")
print(f"    {OUTPUT_LABELED_CSV}")
print(f"    {OUTPUT_METADATA}")

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nDataset: {len(windows_df):,} windows labeled")
print(f"Features per window: {len([c for c in windows_df.columns if c not in ['Time', 'Node', 'Episode_Label']])}")

print(f"\nLabels:")
for label, name in sorted(label_names.items()):
    count = (windows_df['Episode_Label'] == label).sum()
    pct = 100 * count / len(windows_df)
    print(f"  {label} - {name:15s}: {count:>10,} ({pct:>6.2f}%)")

print("\n" + "="*70)
print("NEXT: python create_sequences_simple.py")
print("="*70)