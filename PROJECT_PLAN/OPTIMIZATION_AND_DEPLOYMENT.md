# Optimization & Edge Deployment Specification

## 1. Hyperparameter Optimization (Optuna)
*   **Engine:** Optuna with **TPE Sampler** (Bayesian Optimization).
*   **Search Space:**
    *   `learning_rate`: [1e-4, 1e-2] (Log)
    *   `hidden_dim`: [32, 256] (Int)
    *   `dropout`: [0.1, 0.5] (Float)
    *   `num_layers`: [1, 4] (Int)
*   **Pruning:** Use `MedianPruner` to stop underperforming trials early and save GPU compute.

## 2. Model Compression for Edge AI
To deploy high-complexity models (TFT/Mamba) on the Raspberry Pi:

### Stage 1: PyTorch to ONNX
*   Export models using `torch.onnx.export`.
*   Ensure dynamic axes are set for batch size and sequence length.

### Stage 2: INT8 Quantization
*   Use ONNX Runtime Quantization tools.
*   Apply **Dynamic Quantization** for weights to 8-bit integers.
*   Goal: Reduce model size by 75% and increase inference speed on Pi by 3-5x.

## 3. Edge Execution Environment (Raspberry Pi)
*   **Runtime:** ONNX Runtime (C++ or Python) for maximum efficiency on ARM.
*   **Process Management:** 
    *   **Airflow:** Scheduling hourly ingestion and inference.
    *   **Async Processing:** Ensuring dashboard updates don't block the forecasting loop.
