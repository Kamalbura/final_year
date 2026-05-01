# Software Requirements Specification (SRS)
## Project: SOTA Air Quality Forecasting & Edge AI Deployment

### 1. Introduction
#### 1.1 Purpose
This document specifies the requirements for a comprehensive air quality forecasting platform. It bridges high-end deep learning research with practical Edge AI deployment on low-power hardware (Raspberry Pi).

#### 1.2 Scope
The system will ingest multi-city multivariate air quality and meteorological data, process it using 2025 SOTA standards, train 20+ models across 8 paradigms, and optimize the best performers for real-time edge inference.

### 2. General Description
#### 2.1 Product Perspective
A B.Tech final year project designed for scientific rigor and architectural completeness. It includes a Data Pipeline, a Training Engine (local GPU + Kaggle), and a Deployment Stack (Airflow + Raspberry Pi).

#### 2.2 User Classes and Characteristics
*   **Research Engineer (User):** Manually tunes and trains models using Optuna.
*   **Edge Device (System):** Runs inference on the Raspberry Pi using optimized ONNX models.

#### 2.3 Design and Implementation Constraints
*   **Local Hardware:** RTX 2050 (4GB VRAM) for standard training.
*   **Remote Hardware:** Kaggle (T4/P100) for Transformer/Diffusion models.
*   **Deployment Target:** Raspberry Pi (ARM architecture).

### 3. System Requirements
#### 3.1 Functional Requirements
*   **FR-1: Data Ingestion:** Automated pull from Open-Meteo/Kaggle APIs.
*   **FR-2: Preprocessing:** Implementation of SAITS for imputation and "Probability of Residuals" for outlier detection.
*   **FR-3: Model Training:** Support for 8 generations of models (Statistical to Meta-Learning).
*   **FR-4: Optimization:** Export of PyTorch models to ONNX with INT8 quantization.
*   **FR-5: Dashboard:** Visualization of forecasts and model lineage.

#### 3.2 Non-Functional Requirements
*   **NFR-1: Accuracy:** SOTA models must exceed baseline persistence/ARIMA accuracy by at least 20% (RMSE).
*   **NFR-2: Latency:** Edge inference must complete in < 500ms on a Raspberry Pi.
*   **NFR-3: Reproducibility:** Global seeding and standardized data splits (Chronological 70/15/15).

### 4. Outcomes
*   A 100+ page technical report with standardized plots for 20+ models.
*   A functioning Edge AI device providing real-time local air quality alerts.
*   A comparative study of Transformer vs. Mamba architectures in meteorology.
