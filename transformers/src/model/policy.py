"""
Chessformer Policy Head Implementation

This module implements the policy head for the Chessformer model, which is responsible
for predicting move probabilities from the encoded board representation.

The policy head uses an attention-like mechanism to produce move logits for all possible
moves (from any square to any square) as well as special handling for promotion moves.
"""

import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """
    Policy head for the Chessformer model.

    This module takes the encoded board representation and produces logits for all possible
    chess moves. It uses a query-key mechanism similar to attention to model the relationship
    between source and target squares.

    The output consists of:
    1. Regular move logits: A 64×64 matrix representing source→target square moves
    2. Promotion move logits: A 8×8×4 tensor for source file→target file→promotion piece

    Args:
        embed_dim (int): Dimension of the encoder output embeddings
        promotion_dim (int): Number of promotion piece types (default: 4 for knight, bishop, rook, queen)
    """

    def __init__(self, embed_dim, promotion_dim=4):
        """
        Args:
            embed_dim: Dimension of the encoder output embeddings.
            promotion_dim: Number of promotion piece types (e.g., knight, bishop, rook, queen).
        """
        super(PolicyHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**0.5

        # For handling promotion moves:
        # As described in the paper: "We apply a linear projection to the key vectors
        # representing the promotion rank, generating an additive bias for each possible
        # promotion piece."
        self.promotion_bias = nn.Parameter(
            torch.randn(4, embed_dim)
        )  # 4 promotion piece types

        # Initialize the promotion bias with appropriate scaling
        nn.init.normal_(self.promotion_bias, std=0.02)

    def forward(self, x, legal_moves_mask=None):
        """
        Forward pass through the policy head.

        Creates a distribution over all possible chess moves through a query-key mechanism:
        1. Each square produces a query vector (representing "moving from this square")
        2. Each square produces a key vector (representing "moving to this square")
        3. Dot product between queries and keys produces move logits
        4. For promotion moves, additional biases are applied to moves from the 7th rank to the 8th rank

        Args:
            x: Tensor of shape (B, 64, embed_dim) from the encoder representing the board state
            legal_moves_mask: Optional boolean mask of shape (B, 64, 64) indicating legal moves.
                              Used to mask out illegal moves during inference.

        Returns:
            Tuple of:
                move_logits: Tensor of shape (B, 64, 64) with logits for all regular moves.
                             The dimensions represent (batch, from_square, to_square).
                promotion_logits: Tensor of shape (B, 8, 8, 4) with logits for promotion moves.
                                  The dimensions represent (batch, from_file, to_file, promotion_piece).
        """
        B = x.shape[0]  # Batch size

        # Compute a policy-specific embedding for each square
        policy_emb = self.dense(x)  # (B, 64, embed_dim)

        # Create query vectors (representing "moving from" a square)
        Q = self.query_proj(policy_emb)  # (B, 64, embed_dim)

        # Create key vectors (representing "moving to" a square)
        K = self.key_proj(policy_emb)  # (B, 64, embed_dim)

        # Compute move logits via scaled dot product between queries and keys
        # Higher values indicate higher probability of moving from one square to another
        move_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, 64, 64)

        # --- Handle pawn promotions as described in the paper ---
        # Define indices for the 7th rank (penultimate) and 8th rank (promotion)
        penultimate_rank_indices = torch.arange(
            48, 56, device=x.device
        )  # 7th rank squares
        promotion_rank_indices = torch.arange(
            56, 64, device=x.device
        )  # 8th rank squares

        # Extract keys for the promotion rank
        promotion_rank_keys = K[:, promotion_rank_indices]  # (B, 8, embed_dim)

        # Calculate promotion biases by multiplying promotion rank keys with promotion bias parameter
        # This generates a bias for each possible promotion piece type
        promotion_biases = torch.matmul(
            promotion_rank_keys, self.promotion_bias.t()
        )  # (B, 8, 4)

        # Extract the logits for moves from penultimate rank to promotion rank
        # These are the moves that can result in pawn promotions
        promotion_moves_logits = move_logits[
            :, penultimate_rank_indices.unsqueeze(1), promotion_rank_indices
        ]  # (B, 8, 8)

        # Reshape for adding the promotion biases
        promotion_moves_logits = promotion_moves_logits.unsqueeze(-1)  # (B, 8, 8, 1)
        promotion_biases = promotion_biases.view(B, 1, 8, 4)  # (B, 1, 8, 4)

        # Add biases to create separate logits for each promotion piece type
        # This gives us different logits for promoting to knight, bishop, rook, or queen
        promotion_moves_with_piece_logits = (
            promotion_moves_logits + promotion_biases
        )  # (B, 8, 8, 4)

        # Optional: mask out illegal moves for better inference
        if legal_moves_mask is not None:
            # Mask for regular moves
            move_logits = move_logits.masked_fill(~legal_moves_mask, float("-inf"))

            # Create promotion-specific mask from the provided legal_moves_mask
            promotion_legal_mask = legal_moves_mask[
                :, penultimate_rank_indices.unsqueeze(1), promotion_rank_indices
            ].unsqueeze(-1)
            promotion_legal_mask = promotion_legal_mask.expand(
                -1, -1, -1, 4
            )  # Expand to all promotion piece types

            # Apply mask to promotion moves
            promotion_moves_with_piece_logits = (
                promotion_moves_with_piece_logits.masked_fill(
                    ~promotion_legal_mask, float("-inf")
                )
            )

        return (move_logits, promotion_moves_with_piece_logits)
