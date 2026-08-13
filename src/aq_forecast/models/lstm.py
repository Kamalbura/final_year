"""
lstm.py — LSTM model for AQI time-series forecasting.

Standard unidirectional LSTM with stacked layers, dropout, and
a fully-connected output head. Captures long-range temporal
dependencies via gated memory cells.
"""

import torch
import torch.nn as nn
from src.aq_forecast.config import ModelConfig


class LSTMModel(nn.Module):
    """
    Unidirectional LSTM for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → Stacked LSTM layers with inter-layer dropout
        → Last timestep hidden state
        → FC output head
        → Prediction (batch, horizon)
    """

    def __init__(self, config: ModelConfig, horizon: int = 24):
        super().__init__()
        self.config = config
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_dim)
        Returns:
            predictions: (batch, horizon)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, lookback, hidden_dim)

        # Take the output from the last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)

        out = self.fc(last_output)  # (batch, horizon)
        return out
