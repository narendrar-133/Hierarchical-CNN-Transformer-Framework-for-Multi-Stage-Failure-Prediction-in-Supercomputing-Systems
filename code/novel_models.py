"""
Hybrid CNN-Transformer Model
Novel Contribution: Combines CNN for local pattern extraction with Transformer for temporal modeling
"""

import torch
import torch.nn as nn
import math

class HybridCNNTransformer(nn.Module):
    """
    Novel Architecture: CNN + Transformer Hybrid
    
    Key Innovation:
    1. CNN extracts local patterns within each window (like base paper)
    2. Transformer models temporal dependencies across windows
    3. Multi-head attention for failure pattern recognition
    
    This is novel because:
    - Base paper only used CNN
    - Combines strengths of both architectures
    - Better for sequential log analysis
    """
    
    def __init__(
        self,
        input_dim=18,          # Number of features per window
        cnn_channels=64,       # CNN output channels
        embed_dim=128,         # Transformer embedding dimension
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        seq_len=3
    ):
        super().__init__()
        
        # ===== CNN COMPONENT =====
        # Extracts local patterns from each window's features
        self.cnn = nn.Sequential(
            # 1D convolution over features
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project CNN output to embedding dimension
        self.cnn_projection = nn.Linear(cnn_channels * input_dim, embed_dim)
        
        # ===== TRANSFORMER COMPONENT =====
        # Models temporal dependencies across windows
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # ===== REGRESSION HEAD =====
        # Predicts severity score [0, 3]
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # ===== STEP 1: CNN Feature Extraction =====
        # Process each window independently with CNN
        cnn_outputs = []
        for i in range(seq_len):
            # Get single window: (batch, input_dim)
            window = x[:, i, :]
            
            # Reshape for 1D conv: (batch, 1, input_dim)
            window = window.unsqueeze(1)
            
            # Apply CNN: (batch, cnn_channels, input_dim)
            cnn_out = self.cnn(window)
            
            # Flatten: (batch, cnn_channels * input_dim)
            cnn_out = cnn_out.view(batch_size, -1)
            
            # Project to embedding space: (batch, embed_dim)
            cnn_out = self.cnn_projection(cnn_out)
            
            cnn_outputs.append(cnn_out)
        
        # Stack windows: (batch, seq_len, embed_dim)
        x = torch.stack(cnn_outputs, dim=1)
        
        # ===== STEP 2: Add Positional Encoding =====
        x = x + self.pos_embedding
        
        # ===== STEP 3: Transformer Temporal Modeling =====
        x = self.transformer(x)
        
        # ===== STEP 4: Global Pooling =====
        # Average across sequence
        x = x.mean(dim=1)
        
        # ===== STEP 5: Regression =====
        output = self.regressor(x)
        
        # Clip to [0, 3] range for severity prediction
        return torch.clamp(output.squeeze(-1), 0, 3)


class MultiScaleTransformer(nn.Module):
    """
    Novel Architecture: Multi-Scale Window Transformer
    
    Key Innovation:
    - Processes multiple window sizes simultaneously (3min, 5min, 7min)
    - Captures both short-term and long-term patterns
    - Fusion mechanism to combine multi-scale features
    
    This addresses the base paper's limitation of fixed window size
    """
    
    def __init__(
        self,
        input_dim=18,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        seq_len=3,
        num_scales=3  # Number of different window sizes
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # ===== MULTI-SCALE EMBEDDINGS =====
        # Separate embeddings for each scale
        self.scale_embeddings = nn.ModuleList([
            nn.Linear(input_dim, embed_dim)
            for _ in range(num_scales)
        ])
        
        # ===== POSITIONAL ENCODING =====
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )
        
        # ===== SCALE-SPECIFIC TRANSFORMERS =====
        self.scale_transformers = nn.ModuleList()
        for _ in range(num_scales):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.scale_transformers.append(transformer)
        
        # ===== FUSION MECHANISM =====
        # Attention-based fusion of multi-scale features
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # ===== REGRESSION HEAD =====
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x_multi_scale):
        """
        x_multi_scale: Can be either:
          - List of tensors, one per scale: [(batch, seq_len, input_dim), ...]
          - Single tensor: (batch, seq_len, input_dim) - will be replicated
        """
        # ===== HANDLE INPUT FORMAT =====
        if not isinstance(x_multi_scale, list):
            # Single tensor provided - replicate for all scales
            x_multi_scale = [x_multi_scale for _ in range(self.num_scales)]
        
        # Validate we have enough scales
        if len(x_multi_scale) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {len(x_multi_scale)}")
        
        scale_outputs = []
        
        # ===== STEP 1: Process Each Scale =====
        for scale_idx, x in enumerate(x_multi_scale):
            # Embed
            x = self.scale_embeddings[scale_idx](x)
            
            # Add positional encoding
            x = x + self.pos_embedding
            
            # Transform
            x = self.scale_transformers[scale_idx](x)
            
            # Pool across sequence
            x = x.mean(dim=1)  # (batch, embed_dim)
            
            scale_outputs.append(x)
        
        # ===== STEP 2: Fuse Multi-Scale Features =====
        # Stack: (batch, num_scales, embed_dim)
        scale_outputs = torch.stack(scale_outputs, dim=1)
        
        # Self-attention fusion
        fused, _ = self.fusion_attention(
            scale_outputs, scale_outputs, scale_outputs
        )
        
        # Residual connection
        fused = self.fusion_norm(fused + scale_outputs)
        
        # Pool across scales
        fused = fused.mean(dim=1)  # (batch, embed_dim)
        
        # ===== STEP 3: Regression =====
        output = self.regressor(fused)
        
        # Clip to [0, 3] range
        return torch.clamp(output.squeeze(-1), 0, 3)


class HierarchicalAttentionModel(nn.Module):
    """
    Novel Architecture: Hierarchical Attention
    
    Key Innovation:
    - Window-level attention (which features matter in a window)
    - Sequence-level attention (which windows matter in a sequence)
    - Interpretable attention weights for root cause analysis
    
    Novel contribution: Interpretability for failure prediction
    """
    
    def __init__(
        self,
        input_dim=18,
        hidden_dim=128,
        num_heads=4,
        dropout=0.1,
        seq_len=3
    ):
        super().__init__()
        
        # ===== WINDOW-LEVEL PROCESSING =====
        self.window_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Window-level attention
        self.window_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # ===== SEQUENCE-LEVEL PROCESSING =====
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, hidden_dim)
        )
        
        # Sequence-level attention
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        
        # ===== REGRESSION HEAD =====
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Store attention weights for visualization
        self.last_window_attention = None
        self.last_sequence_attention = None
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # ===== STEP 1: Window-Level Encoding =====
        batch_size, seq_len, input_dim = x.shape
        
        # Encode each window
        window_features = self.window_encoder(x)  # (batch, seq_len, hidden_dim)
        
        # Window-level self-attention
        attended_windows, window_attn_weights = self.window_attention(
            window_features, window_features, window_features
        )
        
        # Store for visualization
        self.last_window_attention = window_attn_weights.detach()
        
        # ===== STEP 2: Sequence-Level Processing =====
        # Add positional encoding
        sequence_features = attended_windows + self.pos_embedding
        
        # Sequence-level self-attention
        attended_sequence, seq_attn_weights = self.sequence_attention(
            sequence_features, sequence_features, sequence_features
        )
        
        # Store for visualization
        self.last_sequence_attention = seq_attn_weights.detach()
        
        # Residual + Norm
        sequence_output = self.sequence_norm(attended_sequence + sequence_features)
        
        # ===== STEP 3: Pooling =====
        # Average pooling
        pooled = sequence_output.mean(dim=1)  # (batch, hidden_dim)
        
        # ===== STEP 4: Regression =====
        output = self.regressor(pooled)
        
        # Clip to [0, 3] range
        return torch.clamp(output.squeeze(-1), 0, 3)
    
    def get_attention_weights(self):
        """Return attention weights for interpretability"""
        return {
            'window_attention': self.last_window_attention,
            'sequence_attention': self.last_sequence_attention
        }


# ===============================
# WEIGHTED LOSS FOR ORDINAL REGRESSION
# ===============================
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for ordinal regression
    
    Severity levels: 0 (Normal), 1 (Early Warning), 2 (Pre-Failure), 3 (Failure)
    Weights handle class imbalance by penalizing errors on rare classes more
    """
    def __init__(self, weight_map):
        super().__init__()
        self.weight_map = weight_map

    def forward(self, preds, targets):
        """
        preds: (batch,) - continuous predictions [0, 3]
        targets: (batch,) - true severity levels [0, 3]
        """
        # Create weight tensor
        weights = torch.zeros_like(targets, dtype=torch.float32)
        for severity, weight in self.weight_map.items():
            weights[targets == severity] = weight
        
        # Weighted MSE
        squared_errors = (preds - targets) ** 2
        weighted_loss = weights * squared_errors
        
        return weighted_loss.mean()


class OrdinalRegressionLoss(nn.Module):
    """
    Alternative: Ordinal Regression Loss
    Specifically designed for ordinal targets (0 < 1 < 2 < 3)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, preds, targets):
        """
        Penalizes predictions based on ordinal distance
        Being off by 2 levels is worse than being off by 1 level
        """
        # Basic MSE
        mse = (preds - targets) ** 2
        
        # Additional penalty for large errors
        ordinal_distance = torch.abs(preds - targets)
        ordinal_penalty = ordinal_distance ** 1.5  # Penalize larger errors more
        
        # Combine
        loss = mse + 0.5 * ordinal_penalty
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - NOT recommended for regression
    This is kept for reference but should use MSE-based losses instead
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, preds, targets):
        # Convert to probabilities
        probs = torch.sigmoid(preds)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            preds, targets, reduction='none'
        )
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
            
        loss = focal_weight * ce_loss
        return loss.mean()