# Raspberry Pi Python Environments Summary

## System Python (3.13.5)
- **Location**: `/usr/bin/python3`
- **Packages**: numpy only
- **Status**: ❌ Missing all ML libraries

## Virtual Environments

### 1. ~/venvs/ml (Python 3.13.5)
- **Status**: ✅ Best for Classical ML
- **Packages**:
  - ✅ scikit-learn 1.8.0
  - ✅ xgboost 3.2.0
  - ✅ joblib 1.5.3
  - ✅ pandas 3.0.1
  - ✅ numpy 2.4.3
  - ❌ lightgbm (MISSING)
  - ❌ catboost (MISSING)
- **Additional**: matplotlib, seaborn, statsmodels, psycopg2-binary, jupyter

### 2. ~/venvs/dl (Python 3.13.5)
- **Status**: ✅ Best for Deep Learning
- **Packages**:
  - ✅ torch 2.10.0
  - ✅ torchvision 0.25.0
  - ✅ torchaudio 2.10.0
  - ✅ tensorflow 2.21.0
  - ✅ keras 3.13.2
  - ✅ pandas 3.0.1
  - ✅ numpy 2.4.3
  - ❌ scikit-learn (MISSING)
  - ❌ xgboost (MISSING)
  - ❌ lightgbm (MISSING)
  - ❌ catboost (MISSING)
  - ❌ joblib (MISSING)
- **Additional**: opencv-python-headless, onnxruntime, jupyter

### 3. ~/venvs/fswd (Python 3.13.5)
- **Status**: Not checked (likely web development)

### 4. ~/venvs/iot (Python 3.13.5)
- **Status**: Not checked (likely IoT/embedded)

### 5. ~/venvs/security (Python 3.13.5)
- **Status**: Not checked (likely security tools)

## Docker Containers

### Airflow (apache/airflow:2.10.3-python3.12)
- **Containers**: scheduler, webserver, triggerer
- **Status**: ❌ Missing ML libraries
- **Packages**:
  - ✅ pandas 2.1.4
  - ✅ numpy 1.26.4
  - ❌ scikit-learn (MISSING)
  - ❌ xgboost (MISSING)
  - ❌ lightgbm (MISSING)
  - ❌ catboost (MISSING)
  - ❌ joblib (MISSING)
- **Additional**: All Airflow providers, cloud SDKs (AWS, GCP, Azure)

### PostgreSQL (postgres:16)
- **Container**: pi_airflow-postgres-1
- **Port**: 5432
- **Status**: ✅ Running and healthy

### Dashboard (node:20-alpine)
- **Container**: pi_airflow-dashboard-1
- **Port**: 3000
- **Status**: ✅ Running (Next.js dashboard)

## Deployment Strategy

### Option 1: Use ml venv (RECOMMENDED)
- Install lightgbm and catboost in ~/venvs/ml
- Copy deployment_models to Pi
- Run inference scripts using ~/venvs/ml/bin/python3
- Integrate with Airflow via BashOperator calling ml venv

### Option 2: Install in Airflow container
- Build custom Airflow image with ML libraries
- Requires rebuilding and redeploying containers
- More complex but better integration

### Option 3: Hybrid approach
- Keep models in ~/projects/final_year/deployment_models/
- Create inference script that uses ml venv
- Airflow DAG calls script via BashOperator with venv activation

## Required Actions

1. Install missing libraries in ml venv:
   ```bash
   source ~/venvs/ml/bin/activate
   pip install lightgbm catboost
   ```

2. Copy deployment models to Pi:
   ```bash
   scp -r deployment_models bura@100.111.13.58:~/projects/final_year/
   ```

3. Test inference with ml venv:
   ```bash
   ssh bura@100.111.13.58 "source ~/venvs/ml/bin/activate && cd ~/projects/final_year/deployment_models && python3 pi_inference.py"
   ```

4. Create Airflow DAG for hourly forecasting
5. Create script to load forecasts into PostgreSQL aq.forecasts table
