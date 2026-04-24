# Research Shortlist for India AQ Forecasting

This shortlist is aligned to your project goal: multivariate hourly/15-minute AQ forecasting for major Indian cities (including Hyderabad).

## Priority Reading (Model Design)

1. Temporal Fusion Transformer for Interpretable Multi-horizon Time Series Forecasting (Lim et al., 2020)
- Why: strongest baseline architecture for multi-horizon forecasting with interpretability.
- Use in this repo: compare your LSTM/TCN/Transformer with TFT-style variable selection and quantile outputs.

2. Predicting PM2.5 levels over Indian metropolitan cities using machine learning approaches (APNet/CNN-LSTM line of work)
- Why: India-specific city data and model behavior.
- Use in this repo: benchmark against PM2.5-focused metro datasets to validate external relevance.

3. Spatio-temporal air quality analysis and PM2.5 prediction over Hyderabad using ML
- Why: directly relevant to your request for Hyderabad-focused modeling.
- Use in this repo: include city-specific error analysis and winter-season bias checks.

4. Transformer-based PM2.5 prediction studies (recent 2023-2025 survey and benchmark papers)
- Why: informs attention-window length, feature engineering, and error characteristics.
- Use in this repo: justify Transformer/RT-Transformer hyperparameters and lag windows.

## Dataset/Protocol Best Practices from Literature

- Use chronological train/val/test split (no random shuffle).
- Report MAE, RMSE, R2, and MAPE/sMAPE when possible.
- Compare city-wise and season-wise performance, not only global average.
- Include persistence baseline and one classical model (e.g., XGBoost/RandomForest).
- Check data completeness and missingness per city before model training.

## Notes for This Repository

- The generated data files in this folder are hourly and city-tagged.
- For your current 15-minute pipeline, either:
  - run training on hourly data directly, or
  - interpolate to 15-minute frequency carefully and flag synthetic intervals.
