"""
bilstm_attention.py — Bidirectional LSTM with Attention for AQI forecasting.

Combines the bidirectional context capture of BiLSTM with an attention
mechanism to focus on the most predictive timesteps in both temporal
directions.
"""

import torch
import torch.nn as nn
from src.aq_forecast.config import ModelConfig
from src.aq_forecast.models.lstm_attention import BahdanauAttention


class BiLSTMAttentionModel(nn.Module):
    """
    BiLSTM + Attention for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → BiLSTM layers (forward + backward concatenated)
        → Attention over all bidirectional hidden states
        → Context vector (2 * hidden_dim)
        → FC output head
        → Prediction (batch, horizon)
    """

    def __init__(self, config: ModelConfig, horizon: int = 24):
        super().__init__()
        self.config = config
        self.horizon = horizon

        self.bilstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attention = BahdanauAttention(
            hidden_dim=config.hidden_dim * 2,
            attention_dim=config.attention_dim,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_dim)
        Returns:
            predictions: (batch, horizon)
        """
        bi_out, (h_n, _) = self.bilstm(x)
        # bi_out: (batch, lookback, hidden_dim * 2)
        # h_n: (num_layers * 2, batch, hidden_dim)

        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2]   # Last layer, forward direction
        h_backward = h_n[-1]  # Last layer, backward direction
        decoder_state = torch.cat([h_forward, h_backward], dim=-1)
        # decoder_state: (batch, hidden_dim * 2)

        context, attn_weights = self.attention(bi_out, decoder_state)
        # context: (batch, hidden_dim * 2)

        context = self.layer_norm(context)
        context = self.dropout(context)

        out = self.fc(context)  # (batch, horizon)
        return out

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization."""
        with torch.no_grad():
            bi_out, (h_n, _) = self.bilstm(x)
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            decoder_state = torch.cat([h_forward, h_backward], dim=-1)
            _, attn_weights = self.attention(bi_out, decoder_state)
        return attn_weights.detach()
