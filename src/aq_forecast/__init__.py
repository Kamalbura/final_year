"""Air-quality forecasting model library.

A self-contained model zoo (RNN, LSTM, BiLSTM, attention variants, TCN,
Transformer, XGBoost, LightGBM) behind a factory, with an Optuna-driven
training loop and AQI-aware evaluation.

This sits alongside the platform code in ``src/`` rather than replacing it:
``src/models/transformers.py`` remains the production encoder used by the
deployed pipeline, while this package is the experimentation surface.
"""
