from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        if model_dim < 1:
            raise ValueError("model_dim must be positive")
        self.dropout = nn.Dropout(dropout)
        self.model_dim = model_dim
        self.max_len = max_len
        self.register_buffer("pe", self._build_encoding(max_len), persistent=False)

    def _build_encoding(self, max_len: int) -> torch.Tensor:
        position = torch.arange(0, max_len).unsqueeze(1)
        pe = torch.zeros(max_len, self.model_dim)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, dtype=torch.float) * (-math.log(10000.0) / self.model_dim)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        if self.model_dim > 1:
            cos_columns = pe[:, 1::2]
            cos_term = torch.cos(position.float() * div_term[: cos_columns.shape[1]])
            pe[:, 1::2] = cos_term
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self.pe = self._build_encoding(seq_len).to(x.device)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ForecastHead(nn.Module):
    def __init__(self, model_dim: int, horizon: int, output_dim: int):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, horizon * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        output = self.proj(x)
        return output.view(batch_size, self.horizon, self.output_dim)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        horizon: int = 16,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim or input_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forecast_head = ForecastHead(model_dim, horizon, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (batch, sequence, features), got {tuple(x.shape)}")
        encoded = self.input_projection(x)
        encoded = self.position_encoding(encoded)
        encoded = self.encoder(encoded)
        summary = encoded[:, -1, :]
        return self.forecast_head(summary)


class RTTransformerForecaster(nn.Module):
    """Lightweight real-time transformer with local causal attention."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        local_window: int = 16,
        horizon: int = 16,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.local_window = local_window
        self.horizon = horizon
        self.output_dim = output_dim or input_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forecast_head = ForecastHead(model_dim, horizon, self.output_dim)

    def _build_local_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        for query_index in range(seq_len):
            left = max(0, query_index - self.local_window + 1)
            mask[query_index, left : query_index + 1] = 0.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (batch, sequence, features), got {tuple(x.shape)}")
        encoded = self.input_projection(x)
        encoded = self.position_encoding(encoded)
        attention_mask = self._build_local_causal_mask(encoded.size(1), encoded.device)
        encoded = self.encoder(encoded, mask=attention_mask)
        summary = encoded[:, -1, :]
        return self.forecast_head(summary)
