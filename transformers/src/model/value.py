import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueHead(nn.Module):
    def __init__(self, embed_dim, d_value=32, value_embedding_dim=128, num_classes=3):
        """
        Args:
            embed_dim: Dimension of the encoder output embeddings.
            d_value: Dimensionality to which each square is projected.
            value_embedding_dim: Dimension of the flattened value embedding.
            num_classes: Number of outcome classes (e.g., win, draw, loss).
        """
        super(ValueHead, self).__init__()
        self.proj1 = nn.Linear(embed_dim, d_value)
        # After projection, we flatten across the 64 board squares.
        self.proj2 = nn.Linear(64 * d_value, value_embedding_dim)
        self.out_proj = nn.Linear(value_embedding_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 64, embed_dim) from the encoder.
        Returns:
            logits: Tensor of shape (B, num_classes) representing game outcome probabilities.
        """
        # Project each token (square) down to d_value dimensions.
        x_val = self.proj1(x)  # (B, 64, d_value)
        # Flatten the board representation.
        x_val_flat = x_val.flatten(start_dim=1)  # (B, 64 * d_value)
        # Project to the value embedding.
        value_embedding = self.proj2(x_val_flat)   # (B, value_embedding_dim)
        # Output the final logits for win/draw/loss.
        logits = self.out_proj(value_embedding)    # (B, num_classes)
        return logits
