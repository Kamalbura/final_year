"""
rnn.py — Vanilla RNN model for AQI time-series forecasting.

Simple Elman RNN with configurable layers, hidden size, and dropout.
Serves as the baseline recurrent architecture.
"""

import torch
import torch.nn as nn
from src.aq_forecast.config import ModelConfig


class RNNModel(nn.Module):
    """
    Vanilla RNN for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → RNN layers with dropout
        → Final hidden state
        → Fully-connected output layer
        → Prediction (batch, horizon)
    """

    def __init__(self, config: ModelConfig, horizon: int = 24) -> None:
        """Initialize the vanilla RNN model.

        Args:
            config: Model hyperparameters (input_dim, hidden_dim,
                num_layers, dropout).
            horizon: Number of future timesteps to forecast.
        """
        super().__init__()
        self.config = config
        self.horizon = horizon

        self.rnn = nn.RNN(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )

        self.dropout = nn.Dropout(config.dropout)

        # Output projection: final hidden → forecast
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
        # RNN forward pass
        rnn_out, h_n = self.rnn(x)
        # rnn_out: (batch, lookback, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)

        # Use the last timestep's output
        last_output = rnn_out[:, -1, :]  # (batch, hidden_dim)
        last_output = self.dropout(last_output)

        # Project to forecast horizon
        out = self.fc(last_output)  # (batch, horizon)
        return out
