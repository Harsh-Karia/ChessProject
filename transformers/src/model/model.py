"""
Chessformer Model Definition

This module implements the Chessformer model architecture, a transformer-based neural network for chess.
The model is designed to process chess positions and predict both move probabilities (policy) and
position evaluation (value).

The architecture consists of:
1. Input Embedding Layer - Converts raw chess features to high-dimensional embeddings
2. Transformer Encoder - Processes embeddings using self-attention to capture piece interactions
3. Policy Head - Predicts move probabilities
4. Value Head - Evaluates the position (win/draw/loss probabilities)
"""

import torch.nn as nn

from .encoder import ChessformerEncoder
from .policy import PolicyHead
from .value import ValueHead


class ChessInputEmbedding(nn.Module):
    """
    Input embedding layer for chess positions.

    This layer converts the raw 112-feature representation of each square
    into a higher-dimensional embedding that's suitable for the transformer.

    The input features include:
    - 8 one-hot vectors (12 features each) for current and past positions (96 features)
    - En passant and castling information
    - Move count since last capture/pawn move/castle (normalized)
    - Repetition flags

    Args:
        embed_dim (int): Dimension of the output embeddings (default: 1024)
    """

    def __init__(self, embed_dim=1024):
        super().__init__()
        # Input features: 112 per square as described in the paper
        # - 8 positions × 12 one-hot piece vectors = 96
        # - En passant and castling info
        # - Moves since last capture/pawn move/castle (normalized)
        # - Repetition flags for 8 positions
        self.input_dim = 112

        # Linear projection to convert 112-dim input to embed_dim
        self.projection = nn.Linear(self.input_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through the embedding layer.

        Args:
            x: Tensor of shape [batch_size, 64, 112] containing the processed chess features
                for each square on the board

        Returns:
            Tensor of shape [batch_size, 64, embed_dim] containing the embedded representations
        """
        # Project the input features to the embedding dimension
        return self.projection(x)


class ChessformerModel(nn.Module):
    """
    Main Chessformer model class.

    This class combines all components of the Chessformer architecture:
    - Input embedding for chess positions
    - Transformer encoder to process the embeddings
    - Policy head to predict move probabilities
    - Value head to evaluate the position

    Args:
        input_dim (int): Dimension of the input features (default: 112)
        embed_dim (int): Dimension of token embeddings (default: 1024)
        num_heads (int): Number of attention heads (default: 32)
        ff_hidden_dim (int): Hidden dimension of feed-forward networks (default: 4096)
        num_layers (int): Number of transformer layers (default: 15)
        d_value (int): Dimension for value projection (default: 32)
        value_embedding_dim (int): Dimension of value embedding (default: 128)
        num_classes (int): Number of value output classes (default: 3 for win/draw/loss)
        dropout (float): Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim=112,
        embed_dim=1024,
        num_heads=32,
        ff_hidden_dim=4096,
        num_layers=15,
        d_value=32,
        value_embedding_dim=128,
        num_classes=3,
        dropout=0.1,
    ):
        super(ChessformerModel, self).__init__()

        # Input embedding layer: converts 112 features per square to embed_dim
        self.input_embedding = ChessInputEmbedding(embed_dim)

        # Transformer encoder: processes the embedded board representation
        self.encoder = ChessformerEncoder(
            embed_dim, num_heads, ff_hidden_dim, num_layers, dropout
        )

        # Policy head: predicts probabilities for each possible move
        self.policy_head = PolicyHead(embed_dim)

        # Value head: evaluates the position (win/draw/loss probabilities)
        self.value_head = ValueHead(
            embed_dim, d_value, value_embedding_dim, num_classes
        )

    def forward(self, x, legal_moves_mask=None):
        """
        Forward pass through the Chessformer model.

        Args:
            x: Input tensor of shape (B, 64, 112) representing chess positions.
               Each position has 64 squares with 112 features per square.
            legal_moves_mask: Optional mask for valid moves (B, 64, 64).
                              Used to mask out illegal moves during inference.

        Returns:
            move_logits: Tuple of (regular_moves, promotion_moves) tensors.
                         regular_moves has shape (B, 64, 64) for source→target square moves.
                         promotion_moves has shape (B, 8, 8, 4) for
                         source file→target file→promotion piece moves.
            value_logits: Tensor of shape (B, num_classes) with position evaluation.
                          Typically (B, 3) for win/draw/loss probabilities.
        """
        # 1. Embed the input: Convert raw features to high-dimensional representation
        x = self.input_embedding(x)  # (B, 64, embed_dim)

        # 2. Process through the transformer encoder
        encoded = self.encoder(x)  # (B, 64, embed_dim)

        # 3. Get move predictions from policy head
        move_logits = self.policy_head(encoded, legal_moves_mask)

        # 4. Get position evaluation from value head
        value_logits = self.value_head(encoded)

        return move_logits, value_logits
