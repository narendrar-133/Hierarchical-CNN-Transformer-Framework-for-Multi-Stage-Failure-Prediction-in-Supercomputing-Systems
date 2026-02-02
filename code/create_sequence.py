"""
SIMPLIFIED: Non-Overlapping Sequence Generation WITHOUT Balancing
Just creates sequences - all balancing handled in training via loss weights
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

# GPU is optional - CuPy not required
GPU_AVAILABLE = False
print("ℹ GPU (CuPy) not used - CPU normalization is fast enough")

# Try Numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("✓ Numba JIT available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not available (optional)")

# ===============================
# CONFIG
# ===============================
SEQUENCE_LENGTH = 3
OVERLAP_STRIDE = SEQUENCE_LENGTH  # NO OVERLAP!

WINDOWS_CSV = "final_labeled_windows.csv"  # Episode-labeled windows
OUTPUT_FEATURES_X = "X_sequences_features.npy"
OUTPUT_FEATURES_Y = "y_labels_features.npy"
OUTPUT_SCALER = "feature_scaler.pkl"
OUTPUT_METADATA = "sequence_metadata.pkl"

FEATURE_COLS = [
    'total_logs', 'error_count', 'fatal_count', 'warning_count',
    'info_count', 'unique_templates', 'unique_components',
    'max_severity', 'avg_severity', 'severity_std',
    'kernel_logs', 'app_logs', 'has_kernel_error', 'log_rate',
    'hour', 'day_of_week', 'is_weekend', 'is_night'
]

print("="*70)
print("NON-OVERLAPPING SEQUENCE GENERATION")
print("="*70)
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Stride: {OVERLAP_STRIDE} (NO OVERLAP)")
print(f"Class balancing: DISABLED (handled in training)")

# ===============================
# NUMBA-OPTIMIZED FUNCTIONS
# ===============================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def create_sequences_numba_no_overlap(feature_matrix, labels_array, node_indices_list, 
                                          node_lengths, seq_length, stride):
        """
        Ultra-fast sequence creation using Numba JIT compilation
        Now with configurable stride (use stride=seq_length for no overlap)
        """
        # Count total sequences first
        total_seqs = 0
        for i in range(len(node_lengths)):
            if node_lengths[i] >= seq_length:
                total_seqs += (node_lengths[i] - seq_length) // stride + 1
        
        # Pre-allocate output arrays
        seq_indices = np.empty((total_seqs, seq_length), dtype=np.int32)
        seq_labels = np.empty(total_seqs, dtype=np.int8)
        
        seq_idx = 0
        start_idx = 0
        
        # Process each node
        for node_i in range(len(node_lengths)):
            node_len = node_lengths[node_i]
            
            if node_len < seq_length:
                start_idx += node_len
                continue
            
            # Get node's indices
            node_inds = node_indices_list[start_idx:start_idx + node_len]
            
            # Create sequences for this node with specified stride
            for i in range(0, node_len - seq_length + 1, stride):
                # Store indices for this sequence
                for j in range(seq_length):
                    seq_indices[seq_idx, j] = node_inds[i + j]
                
                # Store label (from last window - using Episode_Label)
                seq_labels[seq_idx] = labels_array[node_inds[i + seq_length - 1]]
                seq_idx += 1
            
            start_idx += node_len
        
        return seq_indices, seq_labels

    @jit(nopython=True, parallel=True, fastmath=True)
    def extract_sequences(feature_matrix, seq_indices, num_features):
        """Extract actual sequence data using indices"""
        num_seqs = seq_indices.shape[0]
        seq_len = seq_indices.shape[1]
        
        sequences = np.empty((num_seqs, seq_len, num_features), dtype=np.float32)
        
        for i in prange(num_seqs):
            for j in range(seq_len):
                for k in range(num_features):
                    sequences[i, j, k] = feature_matrix[seq_indices[i, j], k]
        
        return sequences

# ===============================
# LOAD DATA
# ===============================
print("\n[1/4] Loading data...")
start_time = datetime.now()

windows_df = pd.read_csv(WINDOWS_CSV)

available_features = [col for col in FEATURE_COLS if col in windows_df.columns]
if len(available_features) < len(FEATURE_COLS):
    missing = set(FEATURE_COLS) - set(available_features)
    print(f"  WARNING: Missing features: {missing}")
    FEATURE_COLS = available_features

print(f"  Loaded {len(windows_df):,} windows from {windows_df['Node'].nunique()} nodes")
print(f"  Using {len(FEATURE_COLS)} features")

# Load metadata if available
try:
    with open("sequence_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    print(f"  Episode metadata loaded")
    print(f"    Episode length: {metadata.get('episode_length', 'N/A')} windows")
    print(f"    Episode stride: {metadata.get('episode_stride', 'N/A')} windows")
except:
    print("  Warning: Could not load episode metadata")

# Check which label column exists
if 'Episode_Label' in windows_df.columns:
    LABEL_COLUMN = 'Episode_Label'
    print("  Using Episode_Label column")
elif 'Final_Label' in windows_df.columns:
    LABEL_COLUMN = 'Final_Label'
    print("  Using Final_Label column")
else:
    raise ValueError("No label column found! Need 'Episode_Label' or 'Final_Label'")

label_counts = windows_df[LABEL_COLUMN].value_counts().sort_index()
label_names = {0: 'Normal', 1: 'Early Warning', 2: 'Pre-Failure', 3: 'Failure'}

# Handle both 3-class and 4-class systems
if len(label_counts) == 3:
    label_names = {0: 'Normal', 1: 'Pre-Failure', 2: 'Failure'}

print("\n  Label distribution:")
for label, count in label_counts.items():
    pct = 100 * count / len(windows_df)
    name = label_names.get(int(label), f'Label {label}')
    print(f"    {name} ({int(label)}): {count:,} ({pct:.2f}%)")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  Time: {elapsed:.1f}s")

# ===============================
# CREATE SEQUENCES (NO OVERLAP)
# ===============================
print(f"\n[2/4] Creating sequences (length={SEQUENCE_LENGTH}, stride={OVERLAP_STRIDE})...")
start_time = datetime.now()

# Sort once
windows_df = windows_df.sort_values(['Node', 'Time']).reset_index(drop=True)

# Extract to numpy
feature_matrix = windows_df[FEATURE_COLS].values.astype(np.float32)
labels_array = windows_df[LABEL_COLUMN].values.astype(np.int8)  # Use detected label column
nodes_array = windows_df['Node'].values

if NUMBA_AVAILABLE:
    print("  Using Numba JIT-compiled sequence creation...")
    
    # Prepare node indices for Numba
    unique_nodes = windows_df['Node'].unique()
    node_indices_list = []
    node_lengths = []
    
    for node in unique_nodes:
        node_mask = nodes_array == node
        node_inds = np.where(node_mask)[0].astype(np.int32)
        node_indices_list.extend(node_inds)
        node_lengths.append(len(node_inds))
    
    node_indices_array = np.array(node_indices_list, dtype=np.int32)
    node_lengths_array = np.array(node_lengths, dtype=np.int32)
    
    # Create sequence indices using Numba
    print("  Computing sequence indices...")
    seq_indices, labels_features = create_sequences_numba_no_overlap(
        feature_matrix, labels_array, node_indices_array,
        node_lengths_array, SEQUENCE_LENGTH, OVERLAP_STRIDE
    )
    
    print(f"  Extracting {len(seq_indices):,} sequences...")
    # Extract actual sequences
    sequences_features = extract_sequences(
        feature_matrix, seq_indices, len(FEATURE_COLS)
    )
    
else:
    # Fallback to standard NumPy (still fast)
    print("  Using vectorized NumPy operations...")
    
    all_sequences = []
    all_labels = []
    
    unique_nodes = windows_df['Node'].unique()
    
    for node_idx, node in enumerate(unique_nodes):
        node_mask = nodes_array == node
        node_indices = np.where(node_mask)[0]
        
        if len(node_indices) < SEQUENCE_LENGTH:
            continue
        
        # Vectorized sequence creation with specified stride
        for start in range(0, len(node_indices) - SEQUENCE_LENGTH + 1, OVERLAP_STRIDE):
            seq_inds = node_indices[start:start + SEQUENCE_LENGTH]
            all_sequences.append(feature_matrix[seq_inds])
            all_labels.append(labels_array[seq_inds[-1]])
        
        if (node_idx + 1) % 500 == 0:
            print(f"    Processed {node_idx+1}/{len(unique_nodes)} nodes...")
    
    sequences_features = np.array(all_sequences, dtype=np.float32)
    labels_features = np.array(all_labels, dtype=np.int8)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  ✓ Created {len(sequences_features):,} sequences in {elapsed:.1f}s")
print(f"  Shape: {sequences_features.shape}")
print(f"  Memory: {sequences_features.nbytes / 1024**2:.1f} MB")

print("\n  Sequence label distribution:")
unique, counts = np.unique(labels_features, return_counts=True)
for label, count in zip(unique, counts):
    pct = 100 * count / len(labels_features)
    name = label_names.get(int(label), f'Label {label}')
    print(f"    {name}: {count:,} ({pct:.2f}%)")

# ===============================
# GPU-ACCELERATED NORMALIZATION
# ===============================
print(f"\n[3/4] Normalizing features...")
start_time = datetime.now()

original_shape = sequences_features.shape
sequences_flat = sequences_features.reshape(-1, original_shape[-1])

# Fit scaler
print("  Computing statistics...")
scaler = StandardScaler()
scaler.fit(sequences_flat)

# Apply normalization on CPU (fast enough for most datasets)
print("  Applying normalization on CPU...")
sequences_features_normalized = scaler.transform(sequences_flat)
sequences_features_normalized = sequences_features_normalized.reshape(original_shape).astype(np.float32)

with open(OUTPUT_SCALER, 'wb') as f:
    pickle.dump(scaler, f)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  ✓ Normalized in {elapsed:.1f}s")

# ===============================
# SAVE DATA
# ===============================
print(f"\n[4/4] Saving sequences...")
start_time = datetime.now()

np.save(OUTPUT_FEATURES_X, sequences_features_normalized)
np.save(OUTPUT_FEATURES_Y, labels_features)

# Save metadata
metadata = {
    'sequence_length': SEQUENCE_LENGTH,
    'stride': OVERLAP_STRIDE,
    'feature_cols': FEATURE_COLS,
    'num_features': len(FEATURE_COLS),
    'label_names': label_names,
    'label_distribution': {
        int(label): int(count) 
        for label, count in zip(*np.unique(labels_features, return_counts=True))
    }
}

with open(OUTPUT_METADATA, 'wb') as f:
    pickle.dump(metadata, f)

elapsed = (datetime.now() - start_time).total_seconds()
print(f"  ✓ Saved in {elapsed:.1f}s:")
print(f"    X: {OUTPUT_FEATURES_X} ({sequences_features_normalized.nbytes / 1024**2:.1f} MB)")
print(f"    y: {OUTPUT_FEATURES_Y} ({labels_features.nbytes / 1024**2:.1f} MB)")
print(f"    Metadata: {OUTPUT_METADATA}")
print(f"    Scaler: {OUTPUT_SCALER}")

# ===============================
# ANALYSIS
# ===============================
print("\n" + "="*70)
print("FINAL ANALYSIS")
print("="*70)

print(f"\nDataset:")
print(f"  Sequences: {len(labels_features):,}")
print(f"  Shape: ({len(labels_features):,}, {SEQUENCE_LENGTH}, {len(FEATURE_COLS)})")
print(f"  Overlap: NO")

print(f"\nFinal Labels:")
unique, counts = np.unique(labels_features, return_counts=True)
for label, count in zip(unique, counts):
    pct = 100 * count / len(labels_features)
    name = label_names.get(int(label), f'Label {label}')
    print(f"  {name}: {count:,} ({pct:.2f}%)")

print(f"\nData quality:")
print(f"  Range: [{sequences_features_normalized.min():.3f}, {sequences_features_normalized.max():.3f}]")
print(f"  Mean: {sequences_features_normalized.mean():.6f}")
print(f"  Std: {sequences_features_normalized.std():.6f}")

print("\n" + "="*70)
print("✓ COMPLETE!")
print("="*70)
print(f"\nFiles created:")
print(f"  - {OUTPUT_FEATURES_X}")
print(f"  - {OUTPUT_FEATURES_Y}")
print(f"  - {OUTPUT_SCALER}")
print(f"  - {OUTPUT_METADATA}")
print("\nNote: Class imbalance will be handled via weighted loss in training")
print("="*70)