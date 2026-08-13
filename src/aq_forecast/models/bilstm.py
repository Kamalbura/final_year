"""
bilstm.py — Bidirectional LSTM for AQI time-series forecasting.

Processes the input sequence in both forward and backward directions,
concatenating hidden states to capture richer temporal context.
"""

import torch
import torch.nn as nn
from src.aq_forecast.config import ModelConfig


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → BiLSTM layers (forward + backward)
        → Concatenated last timestep hidden states (2 * hidden_dim)
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

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)

        # BiLSTM outputs 2x hidden_dim (forward + backward)
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
        bi_out, (h_n, c_n) = self.bilstm(x)
        # bi_out: (batch, lookback, hidden_dim * 2)

        # Last timestep output contains both forward and backward info
        last_output = bi_out[:, -1, :]  # (batch, hidden_dim * 2)
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)

        out = self.fc(last_output)  # (batch, horizon)
        return out
