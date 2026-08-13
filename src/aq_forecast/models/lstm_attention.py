"""
lstm_attention.py — LSTM with Bahdanau (Additive) Attention for AQI forecasting.

Applies a learned attention mechanism over all LSTM hidden states,
allowing the model to dynamically weight which past timesteps
are most relevant for the current prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.aq_forecast.config import ModelConfig


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism.

    Computes attention weights over encoder hidden states conditioned
    on a decoder query state, producing a context vector as a weighted sum.

    Score: s_i = V^T · tanh(W₁ · h_i + W₂ · s)
    where h_i are encoder outputs and s is the decoder state.
    """

    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        """Initializes Bahdanau attention layers.

        Args:
            hidden_dim: Dimensionality of encoder hidden states.
            attention_dim: Dimensionality of the attention projection
                (bottleneck).
        """
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes attention-weighted context vector.

        Args:
            encoder_outputs: Encoder hidden states of shape
                ``(batch, seq_len, hidden_dim)``.
            decoder_state: Decoder query state of shape
                ``(batch, hidden_dim)``.

        Returns:
            context: Attention-weighted sum, ``(batch, hidden_dim)``.
            weights: Attention distribution, ``(batch, seq_len)``.
        """
        # decoder_state: (batch, hidden_dim) → (batch, 1, attention_dim)
        query = self.W2(decoder_state).unsqueeze(1)

        # encoder_outputs: (batch, seq_len, hidden_dim) → (batch, seq_len, attention_dim)
        keys = self.W1(encoder_outputs)

        # Additive scoring: (batch, seq_len, attention_dim)
        energy = torch.tanh(keys + query)
        scores = self.V(energy).squeeze(-1)   # (batch, seq_len)

        # Normalize to get attention weights
        weights = F.softmax(scores, dim=-1)   # (batch, seq_len)

        # Compute context vector
        context = torch.bmm(
            weights.unsqueeze(1),    # (batch, 1, seq_len)
            encoder_outputs,         # (batch, seq_len, hidden_dim)
        ).squeeze(1)                 # (batch, hidden_dim)

        return context, weights


class LSTMAttentionModel(nn.Module):
    """
    LSTM + Bahdanau Attention for sequence-to-value regression.

    Architecture:
        Input (batch, lookback, features)
        → Stacked LSTM layers
        → Attention over all hidden states
        → Context vector
        → FC output head
        → Prediction (batch, horizon)
    """

    def __init__(self, config: ModelConfig, horizon: int = 24) -> None:
        """Initializes LSTM+Attention model.

        Args:
            config: Model configuration with hidden_dim, num_layers,
                dropout, attention_dim, and input_dim.
            horizon: Number of future timesteps to predict.
        """
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

        self.attention = BahdanauAttention(
            hidden_dim=config.hidden_dim,
            attention_dim=config.attention_dim,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

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
        lstm_out, (h_n, _) = self.lstm(x)
        # lstm_out: (batch, lookback, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)

        # Use last layer's hidden state as decoder query
        decoder_state = h_n[-1]  # (batch, hidden_dim)

        # Apply attention over all timestep outputs
        context, attn_weights = self.attention(lstm_out, decoder_state)
        # context: (batch, hidden_dim)

        context = self.layer_norm(context)
        context = self.dropout(context)

        out = self.fc(context)  # (batch, horizon)
        return out

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention weights for visualization.

        Args:
            x: Input tensor of shape ``(batch, lookback, input_dim)``.

        Returns:
            Attention weight tensor of shape ``(batch, seq_len)``
            representing the importance of each input timestep.
        """
        with torch.no_grad():
            lstm_out, (h_n, _) = self.lstm(x)
            decoder_state = h_n[-1]
            _, attn_weights = self.attention(lstm_out, decoder_state)
        return attn_weights.detach()
