"""
EVALUATION SCRIPT - Complete Metrics Analysis
Run this AFTER training to get full evaluation metrics

Loads the saved best model and computes all metrics properly
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

from novel_models import HybridCNNTransformer

# ===============================
# CONFIG
# ===============================
BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("MODEL EVALUATION - COMPLETE METRICS")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===============================
# LOAD DATA
# ===============================
print("\n[1/4] Loading data...")

# Load validation indices
val_idx = torch.load("train_indices_temporal.pt")  # This should be val indices
# Actually load the val indices properly
with open("sequence_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

X = np.load("X_sequences_features.npy")
y = np.load("y_labels_features.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Use last 20% as validation (temporal split)
num_samples = len(X)
split_idx = int(0.8 * num_samples)
X_val = X[split_idx:]
y_val = y[split_idx:]

print(f"  Validation samples: {len(X_val):,}")
print(f"  Sequence shape: {X_val.shape}")

# Get label distribution
unique_labels = torch.unique(y_val).numpy().astype(int)
num_classes = len(unique_labels)
max_label = int(y_val.max())

print(f"  Number of classes: {num_classes}")
print(f"  Classes present: {unique_labels}")

# Determine label names based on number of classes
if num_classes == 3:
    severity_names = {0: 'Normal', 1: 'Pre-Failure', 2: 'Failure'}
elif num_classes == 4:
    severity_names = {0: 'Normal', 1: 'Early Warning', 2: 'Pre-Failure', 3: 'Failure'}
else:
    severity_names = {i: f'Class {i}' for i in range(num_classes)}

print("\n  Label distribution:")
for label in unique_labels:
    count = (y_val == label).sum().item()
    pct = 100 * count / len(y_val)
    name = severity_names.get(label, f'Label {label}')
    print(f"    {name}: {count:,} ({pct:.2f}%)")

# Create data loader
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == "cuda")
)

# ===============================
# LOAD MODEL
# ===============================
print("\n[2/4] Loading best model...")

seq_len = X_val.shape[1]
input_dim = X_val.shape[2]

model = HybridCNNTransformer(
    input_dim=input_dim,
    cnn_channels=64,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    seq_len=seq_len
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load('best_model_CORRECTED.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"  Model loaded from epoch {checkpoint['epoch']}")
print(f"  Best balanced accuracy: {checkpoint['balanced_acc']:.4f}")

# ===============================
# MAKE PREDICTIONS
# ===============================
print("\n[3/4] Making predictions...")

all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        preds = model(xb)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Round predictions for classification
pred_classes = np.round(np.clip(all_preds, 0, max_label)).astype(int)
true_classes = all_targets.astype(int)

print(f"  Predictions shape: {all_preds.shape}")
print(f"  Prediction range: [{all_preds.min():.3f}, {all_preds.max():.3f}]")

# ===============================
# COMPUTE ALL METRICS
# ===============================
print("\n[4/4] Computing metrics...")
print("\n" + "="*70)
print("REGRESSION METRICS")
print("="*70)

mae = mean_absolute_error(all_targets, all_preds)
mse = mean_squared_error(all_targets, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_targets, all_preds)

print(f"\nMean Absolute Error (MAE):     {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Squared Error (MSE):       {mse:.4f}")
print(f"R² Score:                       {r2:.4f}")

# Tolerance analysis
within_025 = np.mean(np.abs(all_preds - all_targets) <= 0.25) * 100
within_050 = np.mean(np.abs(all_preds - all_targets) <= 0.50) * 100
within_075 = np.mean(np.abs(all_preds - all_targets) <= 0.75) * 100
within_100 = np.mean(np.abs(all_preds - all_targets) <= 1.00) * 100

print(f"\nPrediction Accuracy:")
print(f"  Within ±0.25: {within_025:.2f}%")
print(f"  Within ±0.50: {within_050:.2f}%")
print(f"  Within ±0.75: {within_075:.2f}%")
print(f"  Within ±1.00: {within_100:.2f}%")

# ===============================
# PER-CLASS PERFORMANCE
# ===============================
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)

for label in unique_labels:
    mask = true_classes == label
    if mask.sum() == 0:
        continue
    
    class_preds = all_preds[mask]
    class_targets = all_targets[mask]
    
    class_mae = mean_absolute_error(class_targets, class_preds)
    class_rmse = np.sqrt(mean_squared_error(class_targets, class_preds))
    
    mean_pred = class_preds.mean()
    std_pred = class_preds.std()
    
    within_05 = np.sum(np.abs(class_preds - class_targets) <= 0.5)
    within_10 = np.sum(np.abs(class_preds - class_targets) <= 1.0)
    
    pct_05 = 100 * within_05 / mask.sum()
    pct_10 = 100 * within_10 / mask.sum()
    
    print(f"\n{severity_names[label]} (Label {label}):")
    print(f"  Samples:      {mask.sum():>10,}")
    print(f"  MAE:          {class_mae:>10.4f}")
    print(f"  RMSE:         {class_rmse:>10.4f}")
    print(f"  Mean pred:    {mean_pred:>10.4f} (target: {label}.0)")
    print(f"  Std pred:     {std_pred:>10.4f}")
    print(f"  Within ±0.5:  {within_05:>10,} ({pct_05:>5.1f}%)")
    print(f"  Within ±1.0:  {within_10:>10,} ({pct_10:>5.1f}%)")

# ===============================
# CLASSIFICATION METRICS
# ===============================
print("\n" + "="*70)
print("CLASSIFICATION METRICS (Rounded Predictions)")
print("="*70)

# Confusion matrix
cm = confusion_matrix(true_classes, pred_classes, labels=unique_labels)
print("\nConfusion Matrix:")
print("              Predicted")

# Header
header = "Actual  "
for i in unique_labels:
    header += f" {i:>6}"
print(header)

# Rows
for i, label in enumerate(unique_labels):
    row = f"  {label}    "
    for j in range(len(unique_labels)):
        row += f" {cm[i,j]:>6,}"
    print(row)

# Classification report
print("\nClassification Report:")
target_names = [severity_names[i] for i in unique_labels]
print(classification_report(
    true_classes, pred_classes,
    labels=unique_labels,
    target_names=target_names,
    zero_division=0
))

# Balanced accuracy
balanced_acc = balanced_accuracy_score(true_classes, pred_classes)
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print("  (Most important metric for imbalanced data)")

# ===============================
# BINARY DETECTION METRICS
# ===============================
print("\n" + "="*70)
print("BINARY DETECTION (Normal vs Any Warning)")
print("="*70)

# Binary: Normal (0) vs Any Warning (1, 2, ...)
binary_true = (true_classes > 0).astype(int)
binary_pred = (pred_classes > 0).astype(int)

binary_precision = precision_score(binary_true, binary_pred, zero_division=0)
binary_recall = recall_score(binary_true, binary_pred, zero_division=0)
binary_f1 = f1_score(binary_true, binary_pred, zero_division=0)

print(f"\nBinary Classification Metrics:")
print(f"  Precision: {binary_precision:.4f} (When predicting warning, % correct)")
print(f"  Recall:    {binary_recall:.4f} (% of warnings caught)")
print(f"  F1-Score:  {binary_f1:.4f}")

# ===============================
# SAVE RESULTS
# ===============================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results = {
    'regression_metrics': {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'r2': float(r2),
        'within_025': float(within_025),
        'within_050': float(within_050),
        'within_075': float(within_075),
        'within_100': float(within_100)
    },
    'classification_metrics': {
        'balanced_accuracy': float(balanced_acc),
        'binary_precision': float(binary_precision),
        'binary_recall': float(binary_recall),
        'binary_f1': float(binary_f1)
    },
    'confusion_matrix': cm.tolist(),
    'per_class_metrics': {},
    'predictions': all_preds.tolist(),
    'targets': all_targets.tolist(),
    'label_names': severity_names,
    'num_classes': num_classes,
    'evaluation_date': datetime.now().isoformat()
}

# Add per-class metrics
for label in unique_labels:
    mask = true_classes == label
    if mask.sum() > 0:
        class_preds = all_preds[mask]
        class_targets = all_targets[mask]
        
        results['per_class_metrics'][int(label)] = {
            'count': int(mask.sum()),
            'mae': float(mean_absolute_error(class_targets, class_preds)),
            'rmse': float(np.sqrt(mean_squared_error(class_targets, class_preds))),
            'mean_pred': float(class_preds.mean()),
            'std_pred': float(class_preds.std()),
            'within_05_pct': float(100 * np.sum(np.abs(class_preds - class_targets) <= 0.5) / mask.sum()),
            'within_10_pct': float(100 * np.sum(np.abs(class_preds - class_targets) <= 1.0) / mask.sum())
        }

# Save as pickle
with open('complete_evaluation_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✓ Results saved to: complete_evaluation_results.pkl")

# Save as text report
with open('evaluation_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("MODEL EVALUATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Validation samples: {len(y_val):,}\n")
    f.write(f"Number of classes: {num_classes}\n\n")
    
    f.write("REGRESSION METRICS\n")
    f.write("-"*70 + "\n")
    f.write(f"MAE:  {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²:   {r2:.4f}\n\n")
    
    f.write("PREDICTION ACCURACY\n")
    f.write("-"*70 + "\n")
    f.write(f"Within ±0.25: {within_025:.2f}%\n")
    f.write(f"Within ±0.50: {within_050:.2f}%\n")
    f.write(f"Within ±0.75: {within_075:.2f}%\n")
    f.write(f"Within ±1.00: {within_100:.2f}%\n\n")
    
    f.write("CLASSIFICATION METRICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"Binary F1-Score:   {binary_f1:.4f}\n\n")
    
    f.write("PER-CLASS PERFORMANCE\n")
    f.write("-"*70 + "\n")
    for label in unique_labels:
        mask = true_classes == label
        if mask.sum() > 0:
            class_preds = all_preds[mask]
            class_mae = mean_absolute_error(all_targets[mask], class_preds)
            within_05_pct = 100 * np.sum(np.abs(class_preds - all_targets[mask]) <= 0.5) / mask.sum()
            
            f.write(f"\n{severity_names[label]} (Label {label}):\n")
            f.write(f"  Samples:      {mask.sum():,}\n")
            f.write(f"  MAE:          {class_mae:.4f}\n")
            f.write(f"  Mean pred:    {class_preds.mean():.4f}\n")
            f.write(f"  Within ±0.5:  {within_05_pct:.1f}%\n")

print(f"✓ Text report saved to: evaluation_report.txt")

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

print(f"\nModel Performance:")
print(f"  Balanced Accuracy: {balanced_acc:.4f}")
print(f"  MAE:               {mae:.4f}")
print(f"  RMSE:              {rmse:.4f}")
print(f"  R²:                {r2:.4f}")

print(f"\nWarning Detection:")
print(f"  Precision: {binary_precision:.4f}")
print(f"  Recall:    {binary_recall:.4f}")
print(f"  F1-Score:  {binary_f1:.4f}")

print(f"\nPrediction Accuracy:")
print(f"  Within ±0.5: {within_050:.2f}%")
print(f"  Within ±1.0: {within_100:.2f}%")

print("\n" + "="*70)
print("✓ EVALUATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - complete_evaluation_results.pkl (detailed metrics)")
print("  - evaluation_report.txt (human-readable report)")
print("\n" + "="*70)