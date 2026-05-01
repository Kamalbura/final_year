# SOTA Model Architecture Catalog

## Generation 1: Statistical Foundations
*   **ARIMA / SARIMA:** Capturing univariate seasonality.
*   **VAR:** Modeling linear pollutant-weather interactions.

## Generation 2: Core ML Ensembles
*   **Random Forest:** Robust baseline for unscaled data.
*   **XGBoost / CatBoost:** Gradient boosting gold standard.

## Generation 3: Deep Sequence Models
*   **LSTM / GRU:** Handling non-linear long-term dependencies.
*   **Bi-LSTM:** Capturing future/past context for archival analysis.

## Generation 4: Hybrids & Attention
*   **CNN-LSTM:** Local pattern extraction (CNN) + Sequential modeling (LSTM).
*   **Bi-LSTM + Multi-Head Attention:** Visualizing alpha weights to interpret importance of specific hours.

## Generation 5: 2024 SOTA Transformers
*   **Informer / Autoformer:** Long-sequence efficiency.
*   **iTransformer:** Inverted embedding for high-dimensional multivariate correlation.
*   **TFT (Temporal Fusion Transformer):** Multi-horizon quantile forecasting with variable selection.

## Generation 6: State Space Models (SSMs) & Physics-AI
*   **Mamba:** Linear-time sequence modeling (Superior to Transformers in speed/memory).
*   **PINN (Physics-Informed Neural Network):** Custom loss incorporating Advection-Diffusion equations.

## Generation 7: Spatio-Temporal & Generative
*   **ST-GCN:** Graph convolutions for pollutant drift across cities.
*   **Diffusion Models (DST-DDPM):** Probabilistic denoising for stochastic spike modeling.
*   **TimeGPT / Moirai:** Zero-shot foundation model performance comparison.

## Generation 8: Edge & Meta-Learning
*   **Hoeffding Tree:** Online incremental learning on the Pi.
*   **MAML / Reptile:** Few-shot adaptation to new cities (Novelty factor).
