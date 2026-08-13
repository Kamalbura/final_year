"""
transformer.py — Transformer encoder model for AQI time-series forecasting.

Uses positional encoding + multi-head self-attention encoder layers.
Based on Vaswani et al. (2017) "Attention Is All You Need",
adapted for time-series regression (encoder-only architecture).
"""

import math
import torch
import torch.nn as nn
from src.aq_forecast.config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for injecting temporal order
    information into the Transformer input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoder for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → Linear projection to d_model
        → Positional Encoding
        → N × TransformerEncoderLayer (multi-head self-attention + FFN)
        → Global average pooling
        → FC output head
        → Prediction (batch, horizon)

    This is an encoder-only architecture suitable for time-series
    forecasting where we map a fixed input window to a prediction.
    """

    def __init__(self, config: ModelConfig, horizon: int = 24):
        super().__init__()
        self.config = config
        self.horizon = horizon

        d_model = config.transformer_d_model
        nhead = config.transformer_nhead
        num_layers = config.transformer_num_encoder_layers
        dim_ff = config.transformer_dim_feedforward

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.d_model = d_model

        # Input projection: features → d_model
        self.input_projection = nn.Linear(config.input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=config.dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model // 2, horizon),
        )

    def _generate_causal_mask(self, seq_len: int,
                              device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask for self-attention."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_dim)
        Returns:
            predictions: (batch, horizon)
        """
        # Project input features to d_model dimension and scale
        x = self.input_projection(x) * math.sqrt(self.d_model)  # (batch, lookback, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)         # (batch, lookback, d_model)

        # Causal mask: each position can only attend to itself and earlier
        causal_mask = self._generate_causal_mask(x.size(1), x.device)

        # Transformer encoder
        encoded = self.transformer_encoder(x, mask=causal_mask)
        # encoded: (batch, lookback, d_model)

        # Layer norm + global average pooling
        encoded = self.layer_norm(encoded)
        pooled = encoded.mean(dim=1)    # (batch, d_model)

        # Output projection
        out = self.fc(pooled)           # (batch, horizon)
        return out
