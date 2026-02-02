"""
CORRECTED Training Script - All Fixes Applied (Compatible with PyTorch)
Changes from original:
1. Temporal split (no random shuffle)
2. Stable scheduler (ReduceLROnPlateau instead of CosineAnnealing)
3. Reasonable weights (50x instead of 1000x)
4. Lower LR (1e-4 instead of 5e-4)
5. Proper early stopping
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
import pickle
from datetime import datetime

from novel_models import HybridCNNTransformer

# ===============================
# CONFIG
# ===============================
BATCH_SIZE = 1024
EPOCHS = 100
LR = 1e-4  # REDUCED from 5e-4
TRAIN_SPLIT = 0.8
PATIENCE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("CORRECTED TRAINING - ALL FIXES APPLIED")
print("="*70)
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===============================
# FIX #1: REASONABLE WEIGHTS
# ===============================
print("\n" + "="*70)
print("FIX #1: REASONABLE CLASS WEIGHTS (was 1000x, now 50x)")
print("="*70)

# OLD (caused instability):
# WEIGHT_MAP = {0: 1.0, 1: 1000.0, 2: 1000.0, 3: 30.0}

# NEW (stable):
WEIGHT_MAP = {
    0: 1.0,      # Normal (96.28%)
    1: 50.0,     # Early Warning (0.01%) - reduced from 1000!
    2: 25.0,     # Pre-Failure (0.05%) - reduced from 1000!
    3: 5.0       # Failure (3.66%) - reduced from 30!
}

print("\nClass weights:")
severity_names = {0: 'Normal', 1: 'Early Warning', 2: 'Pre-Failure', 3: 'Failure'}
for severity, weight in WEIGHT_MAP.items():
    print(f"  {severity_names[severity]:20s} (Label {severity}): weight = {weight:>6.1f}")

# ===============================
# WEIGHTED LOSS
# ===============================
class WeightedMSELoss(nn.Module):
    def __init__(self, weight_map):
        super().__init__()
        self.weight_map = weight_map

    def forward(self, preds, targets):
        weights = torch.zeros_like(targets, dtype=torch.float32)
        for severity, weight in self.weight_map.items():
            weights[targets == severity] = weight
        
        squared_errors = (preds - targets) ** 2
        weighted_loss = weights * squared_errors
        
        return weighted_loss.mean()

# ===============================
# LOAD DATA
# ===============================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

with open("sequence_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
    
print(f"\nMetadata:")
print(f"  Sequence length: {metadata['sequence_length']}")
print(f"  Stride: {metadata['stride']}")

X = np.load("X_sequences_features.npy")
y = np.load("y_labels_features.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

num_samples = X.shape[0]
seq_len = X.shape[1]
input_dim = X.shape[2]

print(f"\nData loaded:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Distribution
print("\nLabel distribution:")
unique, counts = torch.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    pct = 100 * count / num_samples
    print(f"  {severity_names[int(label)]:20}: {count:>10,} ({pct:>6.2f}%)")

# ===============================
# FIX #2: TEMPORAL SPLIT (NO SHUFFLE)
# ===============================
print("\n" + "="*70)
print("FIX #2: TEMPORAL TRAIN/VAL SPLIT (NO RANDOM SHUFFLE)")
print("="*70)

print("\nUsing TEMPORAL split:")
print("  Training: First 80% chronologically")
print("  Validation: Last 20% chronologically")
print("  Tests FUTURE prediction ability!")

split_idx = int(TRAIN_SPLIT * num_samples)

# NO SHUFFLING - keep time order
train_idx = torch.arange(0, split_idx)
val_idx = torch.arange(split_idx, num_samples)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print(f"\nTrain: {len(X_train):,} samples (indices 0 to {split_idx-1})")
print(f"Val:   {len(X_val):,} samples (indices {split_idx} to {num_samples-1})")

# Check distributions
print("\nTrain distribution:")
for label, count in zip(*torch.unique(y_train, return_counts=True)):
    pct = 100 * count / len(y_train)
    print(f"  {severity_names[int(label)]:20}: {count:>10,} ({pct:>6.2f}%)")

print("\nVal distribution:")
for label, count in zip(*torch.unique(y_val, return_counts=True)):
    pct = 100 * count / len(y_val)
    print(f"  {severity_names[int(label)]:20}: {count:>10,} ({pct:>6.2f}%)")

# Save indices
torch.save(train_idx, "train_indices_temporal.pt")
torch.save(val_idx, "val_indices_temporal.pt")
print("\n✓ Split indices saved")

# Data loaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,  # Shuffle within training is OK
    pin_memory=(DEVICE == "cuda")
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,  # Never shuffle validation
    pin_memory=(DEVICE == "cuda")
)

print("✓ Data loaders created")

# ===============================
# MODEL
# ===============================
print("\n" + "="*70)
print("MODEL INITIALIZATION")
print("="*70)

model = HybridCNNTransformer(
    input_dim=input_dim,
    cnn_channels=64,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.2,  # Increased from 0.1
    seq_len=seq_len
).to(DEVICE)

print(f"\nModel: Hybrid CNN-Transformer")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===============================
# FIX #3: STABLE SCHEDULER
# ===============================
print("\n" + "="*70)
print("FIX #3: STABLE SCHEDULER (ReduceLROnPlateau)")
print("="*70)

criterion = WeightedMSELoss(WEIGHT_MAP)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01
)

# OLD (caused collapse):
# scheduler = CosineAnnealingWarmRestarts(...)

# NEW (stable) - REMOVED verbose parameter
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # Maximize balanced accuracy
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

print(f"\n✓ Optimizer: AdamW (lr={LR})")
print(f"✓ Scheduler: ReduceLROnPlateau (no restarts, stable)")
print(f"✓ Loss: WeightedMSELoss")

# ===============================
# TRAINING FUNCTIONS
# ===============================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    pred_classes = np.round(np.clip(all_preds, 0, 3)).astype(int)
    true_classes = all_targets.astype(int)
    
    balanced_acc = balanced_accuracy_score(true_classes, pred_classes)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return {
        'loss': total_loss / len(val_loader),
        'balanced_acc': balanced_acc,
        'mae': mae,
        'preds': all_preds,
        'targets': all_targets
    }


# ===============================
# TRAINING LOOP
# ===============================
print("\n" + "="*70)
print("TRAINING (STABLE, NO COLLAPSE)")
print("="*70)

best_balanced_acc = 0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_balanced_acc': [], 'val_mae': []}

for epoch in range(1, EPOCHS + 1):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Validate
    val_results = validate(model, val_loader, criterion, DEVICE)
    
    # Update scheduler with balanced accuracy
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_results['balanced_acc'])
    new_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_results['loss'])
    history['val_balanced_acc'].append(val_results['balanced_acc'])
    history['val_mae'].append(val_results['mae'])
    
    # Print
    print(f"Epoch [{epoch:3d}/{EPOCHS}] | "
          f"Loss: {train_loss:.4f}/{val_results['loss']:.4f} | "
          f"MAE: {val_results['mae']:.4f} | "
          f"Bal.Acc: {val_results['balanced_acc']:.4f} | "
          f"LR: {new_lr:.6f}")
    
    if old_lr != new_lr:
        print(f"  ↓ LR reduced: {old_lr:.6f} → {new_lr:.6f}")
    
    # Save best
    if val_results['balanced_acc'] > best_balanced_acc:
        best_balanced_acc = val_results['balanced_acc']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'balanced_acc': best_balanced_acc,
            'mae': val_results['mae']
        }, 'best_model_CORRECTED.pth')
        print(f"  ✓ Best model saved (Bal.Acc: {best_balanced_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⚠ Early stopping at epoch {epoch}")
        break
    
    # Check for collapse
    if val_results['loss'] > 10:
        print(f"\n⚠ Loss exploded ({val_results['loss']:.2f})")
        break

print(f"\n✓ Training complete!")
print(f"  Best Balanced Accuracy: {best_balanced_acc:.4f}")

# Save history
with open('training_history_CORRECTED.pkl', 'wb') as f:
    pickle.dump(history, f)

# ===============================
# FINAL EVALUATION
# ===============================
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

checkpoint = torch.load('best_model_CORRECTED.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

val_results = validate(model, val_loader, criterion, DEVICE)
all_preds = val_results['preds']
all_targets = val_results['targets']

pred_classes = np.round(np.clip(all_preds, 0, 3)).astype(int)
true_classes = all_targets.astype(int)

mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
r2 = r2_score(all_targets, all_preds)

print(f"\nRegression Metrics:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# Per-class
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)

for label in [0, 1, 2, 3]:
    mask = true_classes == label
    if mask.sum() > 0:
        class_preds = all_preds[mask]
        class_targets = all_targets[mask]
        class_mae = mean_absolute_error(class_targets, class_preds)
        within_05 = np.mean(np.abs(class_preds - class_targets) <= 0.5) * 100
        
        print(f"\n{severity_names[label]} (Label {label}):")
        print(f"  Samples:      {mask.sum():>10,}")
        print(f"  MAE:          {class_mae:>10.4f}")
        print(f"  Mean pred:    {class_preds.mean():>10.4f} (target: {label}.0)")
        print(f"  Within ±0.5:  {int(mask.sum() * within_05/100):>10,} ({within_05:>5.1f}%)")

# Confusion matrix
print("\n" + "="*70)
print("CLASSIFICATION METRICS")
print("="*70)

cm = confusion_matrix(true_classes, pred_classes)
print("\nConfusion Matrix:")
print("              Predicted")
print("Actual   0     1     2     3")
for i, row in enumerate(cm):
    print(f"  {i}    {row[0]:>6,} {row[1]:>5,} {row[2]:>5,} {row[3]:>5,}")

print("\nClassification Report:")
print(classification_report(
    true_classes, pred_classes,
    target_names=list(severity_names.values()),
    zero_division=0
))

balanced_acc = balanced_accuracy_score(true_classes, pred_classes)
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

# Save results
results = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'balanced_acc': balanced_acc,
    'confusion_matrix': cm,
    'predictions': all_preds,
    'targets': all_targets,
    'history': history
}

with open('final_results_CORRECTED.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n✓ Results saved to final_results_CORRECTED.pkl")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nKey Results:")
print(f"  Balanced Accuracy: {balanced_acc:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²: {r2:.4f}")
print("="*70)