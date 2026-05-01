# Validation & Reporting Protocol

## 1. Metric Suite
### 1.1 Point Forecast Metrics
*   **RMSE (Root Mean Squared Error):** Primary scale-dependent metric.
*   **MAE (Mean Absolute Error):** Robust to outliers.
*   **R² (Coefficient of Determination):** Explaining variance.
*   **MASE (Mean Absolute Scaled Error):** Comparing against a "naive" (Persistence) baseline.

### 1.2 Probabilistic Forecast Metrics (TFT/Diffusion)
*   **Pinball Loss:** For specific quantiles (10th, 50th, 90th).
*   **CRPS (Continuous Ranked Probability Score):** Assessing full distribution accuracy.

## 2. Standardized Reporting Artifacts
Each trained model must produce the following for the B.Tech report:

1.  **Convergence Plot:** Epoch vs. Train/Val Loss.
2.  **Parity Plot:** Scatter plot of Predicted vs. Actual values.
3.  **Timeseries Forecast Plot:** 7-day test set prediction overlay.
4.  **Error Histogram:** Distribution of residuals.
5.  **Optuna Parallel Coordinates:** Visualizing hyperparameter importance.

## 3. Scientific Comparison Matrix
The final report will conclude with a massive table comparing:
*   Inference Time (ms)
*   Model Size (MB)
*   RMSE on Test Set
*   $R^2$ Score
*   Top-3 influential features (via SHAP or Attention Weights).
