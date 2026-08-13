"""
Model registry — factory function to instantiate any model by name.
"""

from src.aq_forecast.config import ModelConfig
from src.aq_forecast.models.rnn import RNNModel
from src.aq_forecast.models.lstm import LSTMModel
from src.aq_forecast.models.bilstm import BiLSTMModel
from src.aq_forecast.models.lstm_attention import LSTMAttentionModel
from src.aq_forecast.models.bilstm_attention import BiLSTMAttentionModel
from src.aq_forecast.models.tcn import TCNModel
from src.aq_forecast.models.transformer import TransformerModel
from src.aq_forecast.models.xgboost_model import XGBoostModel
from src.aq_forecast.models.lightgbm_model import LightGBMModel


# Map model names to their classes
MODEL_REGISTRY = {
    "RNN": RNNModel,
    "LSTM": LSTMModel,
    "BiLSTM": BiLSTMModel,
    "LSTM_Attention": LSTMAttentionModel,
    "BiLSTM_Attention": BiLSTMAttentionModel,
    "TCN": TCNModel,
    "Transformer": TransformerModel,
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
}


def build_model(model_name: str, config: ModelConfig,
                horizon: int = 24, random_state: int = 42):
    """
    Factory function to create a model by name.

    Args:
        model_name: One of ALL_MODELS from config.
        config: ModelConfig with architecture hyperparameters.
        horizon: Forecast horizon (for DL models).
        random_state: Random seed (for GB models).

    Returns:
        Instantiated model (nn.Module or GB wrapper).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_name]

    if model_name in ("XGBoost", "LightGBM"):
        return cls(config=config, random_state=random_state)
    else:
        return cls(config=config, horizon=horizon)
