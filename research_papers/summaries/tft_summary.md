# Research Paper Summary: Temporal Fusion Transformer (TFT)

## Citation
Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

## Key Technical Contributions
1.  **Multi-Horizon Forecasting:** Specifically designed to predict multiple time steps ahead (e.g., next 24 hours) rather than just the next single step.
2.  **Variable Selection Networks (VSN):** Automatically identifies and prioritizes the most important features (pollutants vs. weather) at each time step.
3.  **Gated Residual Networks (GRN):** Provides adaptive depth by allowing the model to skip unnecessary non-linear transformations for simpler data patterns.
4.  **Interpretable Self-Attention:** Uses a specialized attention mechanism that allows humans to visualize which past time steps were most influential for a specific forecast.
5.  **Quantile Regression:** Outputs predictive intervals (e.g., 10th, 50th, 90th percentiles), providing a measure of uncertainty rather than just a point forecast.

## Relevance to this Project
The TFT serves as the "Interpretable SOTA" benchmark in Phase 4. It allows us to explain to stakeholders (and examiners) *why* a specific PM2.5 spike was predicted (e.g., high humidity 3 hours prior).
