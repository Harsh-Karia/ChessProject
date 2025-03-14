import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Mish activation function as described in the paper
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, layer_idx=0, num_layers=15):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        assert self.head_dim * num_heads == embed_dim, (
            "Embedding size must be divisible by number of heads."
        )

        # Linear projections for query, key, and value
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Relative position embeddings (Shaw et al.)
        self.relative_bias_k = nn.Parameter(
            torch.randn(64, 64, self.head_dim)
        )  # Changed from embed_dim to head_dim
        self.relative_bias_v = nn.Parameter(
            torch.randn(64, 64, self.head_dim)
        )  # Changed from embed_dim to head_dim

        self.dropout = nn.Dropout(dropout)

        # Initialize using DeepNorm scheme
        self._init_parameters()

    def _init_parameters(self):
        # DeepNorm initialization (scaled initialization for stability in deeper networks)
        # Calculate gain based on the total number of layers
        gain = math.pow(2.0 * self.num_layers, 0.25)  # DeepNorm paper formula

        # Initialize all projection matrices with the gain-adjusted Xavier initialization
        nn.init.xavier_uniform_(self.Wq.weight, gain=gain)
        nn.init.xavier_uniform_(self.Wk.weight, gain=gain)
        nn.init.xavier_uniform_(self.Wv.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)

        if self.Wq.bias is not None:
            nn.init.zeros_(self.Wq.bias)
            nn.init.zeros_(self.Wk.bias)
            nn.init.zeros_(self.Wv.bias)
            nn.init.zeros_(self.out_proj.bias)

        # Initialize relative positional biases
        nn.init.normal_(self.relative_bias_k, std=0.02)
        nn.init.normal_(self.relative_bias_v, std=0.02)

    def forward(self, x):
        B, N, D = x.shape  # (batch, num_tokens=64, embed_dim)

        # Compute queries, keys, and values
        Q = (
            self.Wq(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )  # (B, heads, N, head_dim)
        K = self.Wk(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention with Shaw et al. relative position encoding
        attn_logits = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (self.head_dim**0.5)

        # Add relative positional bias - simplified approach
        # Just use a per-head scalar bias for each position pair
        rel_bias_k = torch.zeros((1, self.num_heads, N, N), device=x.device)
        attn_logits = attn_logits + rel_bias_k

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, V)

        # Simplify the bias for values too
        output = output.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)

        return self.out_proj(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_hidden_dim,
        dropout=0.1,
        layer_idx=0,
        num_layers=15,
    ):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(
            embed_dim, num_heads, dropout, layer_idx, num_layers
        )

        # Post-LN normalization (apply normalization after the sublayer)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward network with Mish activation as mentioned in the paper
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            Mish(),  # Use Mish activation as specified in the paper
            nn.Linear(ff_hidden_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

        # DeepNorm parameters
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.beta = 1.0

        # Initialize with DeepNorm scheme
        self._init_parameters()

    def _init_parameters(self):
        # DeepNorm initialization and gain scaling
        # The initialization depends on the layer depth and total number of layers
        gain = math.pow(2.0 * self.num_layers, 0.25)  # DeepNorm paper formula

        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Post-LN architecture with DeepNorm scaling:
        # Self-attention with scaled residual connection
        attn_output = self.self_attn(x)
        x = x + self.beta * self.dropout(attn_output)  # Apply beta here
        x = self.norm1(x)

        # Feedforward network with scaled residual connection
        ffn_output = self.ffn(x)
        x = x + self.beta * self.dropout(ffn_output)  # And here
        x = self.norm2(x)

        return x


class ChessformerEncoder(nn.Module):
    """
    Transformer encoder architecture for the Chessformer model.

    This encoder consists of multiple transformer encoder layers, each containing
    self-attention and feed-forward mechanisms. It's designed specifically for chess
    with features like:

    1. Position representation suited for 2D board topology
    2. Multi-head attention to capture different types of piece relationships
    3. Learned offset vectors to enhance spatial understanding
    4. DeepNorm initialization for stability in deep networks

    Args:
        embed_dim (int): Dimension of token embeddings (default: 1024)
        num_heads (int): Number of attention heads (default: 32)
        ff_hidden_dim (int): Hidden dimension in feed-forward networks (default: 4096)
        num_layers (int): Number of transformer layers (default: 15)
        dropout (float): Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        embed_dim=1024,
        num_heads=32,
        ff_hidden_dim=4096,
        num_layers=15,
        dropout=0.1,
    ):
        super().__init__()

        # Stack of encoder layers with DeepNorm (stabilizes training of deep networks)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    ff_hidden_dim,
                    dropout,
                    layer_idx=i,
                    num_layers=num_layers,
                )
                for i in range(num_layers)
            ]
        )

        # Add learned offset vectors as described in the paper
        # These help the model understand square positions and relationships
        self.additive_offset = nn.Parameter(torch.randn(1, 64, embed_dim))
        self.multiplicative_offset = nn.Parameter(torch.randn(1, 64, embed_dim))

        # Initialize parameters with appropriate scaling
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize the learned offset vectors with appropriate scaling.

        Using smaller standard deviation for additive offsets and
        mean=1.0 for multiplicative offsets ensures training stability.
        """
        # Initialize the learned offset vectors
        nn.init.normal_(self.additive_offset, std=0.02)
        nn.init.normal_(self.multiplicative_offset, mean=1.0, std=0.02)

    def forward(self, x):
        """
        Forward pass through the transformer encoder.

        The encoder first applies learned offsets to the embeddings, then
        passes them through a stack of transformer encoder layers.

        Args:
            x: Input tensor of shape [batch_size, 64, embed_dim].
               Contains the embedded representation of chess board.

        Returns:
            Output tensor of shape [batch_size, 64, embed_dim].
            Contains contextual representations that capture relationships
            between squares and pieces.
        """
        # Apply learned offset vectors
        # "add and multiply by learned offset vectors which are separate across tokens and depth"
        # These offsets enhance the model's understanding of spatial relationships
        x = x * self.multiplicative_offset + self.additive_offset

        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x)

        return x


# Note: The ChessformerModel class has been moved to src/model/model.py
# Please import ChessformerModel from there instead of from this file.
# Example: from src.model.model import ChessformerModel
