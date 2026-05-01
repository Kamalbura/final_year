# Research Paper Summary: Mamba (Selective State Spaces)

## Citation
Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *Proceedings of the International Conference on Machine Learning (ICML)*.

## Key Technical Contributions
1.  **Selective State Space Model (SSM):** Introduces a selection mechanism that allows the model to "remember" or "forget" information based on the input, effectively handling the long-range dependency problem without attention.
2.  **Linear Complexity:** Unlike Transformers, which have $O(L^2)$ complexity (quadratic with sequence length), Mamba scales linearly $O(L)$. This makes it much more efficient for very long lookback windows.
3.  **Hardware-Aware Algorithm:** Uses a "Scan" operation optimized for modern GPUs (SRAM vs DRAM management), allowing for faster training than previous SSMs.
4.  **Recurrence-Convolution Dualism:** Mamba can be computed like a Convolution (for fast training) and like a Recurrence (for fast, constant-time inference at the Edge).

## Relevance to this Project
Mamba represents the "Inference Efficiency" SOTA. In our report, we use it to prove that we can achieve Transformer-level accuracy with significantly lower latency on the Raspberry Pi.
