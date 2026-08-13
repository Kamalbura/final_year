"""
tcn.py — Temporal Convolutional Network for AQI time-series forecasting.

Implements a TCN with causal dilated convolutions, residual connections,
and weight normalization. TCNs offer parallelizable training,
flexible receptive fields, and stable gradients compared to RNNs.

Based on: Bai et al. (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling"
"""

import logging

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List

from src.aq_forecast.config import ModelConfig

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """
    Causal convolution: ensures output at time t only depends on
    inputs at time t and earlier. Uses left-padding to maintain
    sequence length.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len) — same length, causal
        """
        out = self.conv(x)
        # Remove right-side padding to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Single TCN residual block with two causal convolutions.

    Architecture:
        Input → CausalConv1d → ReLU → Dropout
              → CausalConv1d → ReLU → Dropout
              + Residual connection (with 1x1 conv if channel mismatch)
              → Output
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels,
                                  kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels,
                                  kernel_size, dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection — downsample if channels differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → Transpose to (batch, features, lookback)
        → Stacked TemporalBlocks with exponentially increasing dilation
        → Global average pooling over temporal dimension
        → FC output head
        → Prediction (batch, horizon)

    The receptive field grows exponentially with depth:
        RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
    """

    def __init__(self, config: ModelConfig, horizon: int = 24):
        super().__init__()
        self.config = config
        self.horizon = horizon

        num_channels = config.tcn_num_channels  # e.g., [64, 64, 64, 64]
        kernel_size = config.tcn_kernel_size

        layers = []
        num_levels = len(num_channels)
        in_channels = config.input_dim

        for i in range(num_levels):
            dilation = 2 ** i
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                dilation=dilation, dropout=config.dropout
            ))
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

        # Validate receptive field covers lookback window
        dilations = [2 ** i for i in range(num_levels)]
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        lookback = 72  # default lookback window
        if self.receptive_field < lookback:
            logger.warning(
                "TCN receptive field (%d) < lookback window (%d). "
                "Increase num_channels (more blocks) or kernel_size.",
                self.receptive_field, lookback,
            )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(num_channels[-1] // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_dim)
        Returns:
            predictions: (batch, horizon)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, lookback)

        out = self.network(x)   # (batch, last_channel, lookback)

        # Global average pooling over temporal dimension
        out = out.mean(dim=-1)  # (batch, last_channel)

        out = self.fc(out)      # (batch, horizon)
        return out
