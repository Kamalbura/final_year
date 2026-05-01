# Training Pipeline Analysis & Fixes

## 🔍 Issues Found (After 5 Hours of Running)

### 1. CatBoost Completely Stuck
- **Status**: Started at 05:19, still running at 09:42 (4h 23min)
- **Problem**: No output directory created - stuck in Optuna hyperparameter search
- **Memory**: Using 846MB RAM but making no progress
- **Root Cause**: CatBoost with Optuna (20 trials) can be extremely slow on large datasets

### 2. Plot Quality Issues Discovered

#### Random Forest (Best model so far):
- **RMSE**: 15.58, **R²**: 0.572
- **Timeseries Plot**: ❌ **Severe underestimation**
  - Actual peaks: ~160 US AQI
  - Predicted peaks: ~140 US AQI
  - Missing ~20 points on peaks!
  
- **Error Histogram**: ❌ **Systematic negative bias**
  - Peak of distribution is LEFT of zero
  - Model consistently underpredicts
  - Should be centered on zero for good model

#### XGBoost:
- **RMSE**: 17.60, **R²**: 0.453
- Similar patterns - underestimating peaks

### 3. Insufficient Logging
- No per-trial progress from Optuna
- Can't tell if models are progressing or stuck
- No warning about slow models

---

## ✅ Fixes in v2 Pipeline

### File: `scripts/unified_training_pipeline_v2.py`

### Key Improvements:

1. **Detailed Optuna Logging**
   ```python
   # NEW: Progress callback shows every trial
   [XGBoost] Trial 5/20: NEW BEST = 0.1234, params={'max_depth': 6, ...}
   [XGBoost] Trial 10/20: current=0.1456, best=0.1234
   ```

2. **CatBoost Timeout & Limits**
   ```python
   # Reduced iterations: 50-200 (was 100-500)
   # Reduced depth: 4-8 (was 4-10)
   # Smaller subset for Optuna: 400 samples
   # 10-minute timeout per phase
   ```

3. **Skip Problematic Models**
   ```bash
   # NEW: Skip CatBoost flag
   python scripts/unified_training_pipeline_v2.py --skip-catboost
   ```

4. **Pause Between Models**
   ```bash
   # NEW: Interactive mode
   python scripts/unified_training_pipeline_v2.py --pause
   ```

5. **Better Console Output**
   - Clear progress per model
   - Time estimates
   - Final ranking table

---

## 🚀 How to Run (Fixed Version)

### Option 1: RECOMMENDED - Skip CatBoost
```powershell
conda run -n dl-env python scripts/unified_training_pipeline_v2.py --skip-catboost
```
**Time**: ~30-45 minutes for 7 models

### Option 2: Interactive with Pauses
```powershell
conda run -n dl-env python scripts/unified_training_pipeline_v2.py --skip-catboost --pause
```
**Time**: ~30-45 minutes (with pauses to check progress)

### Option 3: Windows Batch Menu
```powershell
.\run_training_v2.bat
```
Then select option 1

---

## 📊 Current Best Results (From Previous Run)

| Rank | Model | RMSE | R² | Time | Status |
|------|-------|------|-----|------|--------|
| 🥇 | **Random Forest** | **15.58** | **0.572** | 32 min | ✅ Good |
| 🥈 | **SVR** | 17.17 | 0.480 | 3 min | ✅ Good |
| 🥉 | **XGBoost** | 17.60 | 0.453 | 37 min | ✅ Good |
| 4 | LightGBM | 20.97 | 0.223 | 7 min | ✅ OK |
| 5 | VAR | 39.61 | -1.700 | <1 min | ⚠️ Poor |
| 6 | ARIMA | 55.22 | -4.250 | <1 min | ⚠️ Poor |
| 7 | SARIMA | 63.64 | -5.972 | 10 min | ⚠️ Poor |
| - | CatBoost | - | - | 4h+ | ❌ **STUCK** |

---

## 🎯 Recommendations

1. **Use v2 pipeline** with `--skip-catboost`
2. **Check plots immediately** after each model completes
3. **Focus on Random Forest** - best performance so far
4. **Consider XGBoost** - good balance of speed/accuracy

---

## 🖼️ Plot Quality Check

After running v2, verify these in `outputs/hyderabad/{model}/plots/`:

- ✅ **parity.png**: Points should cluster around red diagonal line
- ⚠️ **timeseries.png**: Check if predicted (orange) follows actual (blue) peaks
- ⚠️ **error_histogram.png**: Should be centered near zero (not shifted left/right)

---

## ⏱️ Expected Timeline (v2 with --skip-catboost)

| Phase | Models | Est. Time |
|-------|--------|-----------|
| Statistical | ARIMA, SARIMA, VAR | ~15 seconds |
| Classical ML | SVR, RF, XGB, LGBM | ~25-40 minutes |
| **Total** | **7 models** | **~30-45 min** |

---

## 🔧 Troubleshooting

**If a model is stuck >30 minutes:**
1. Press `Ctrl+C` to cancel
2. Rerun with `--skip-catboost`
3. Or train models individually: `--models XGBoost Random_Forest`

**If plots look wrong:**
1. Check data scaling in `metrics.json`
2. Verify RMSE/MAE values are reasonable (<50 for Hyderabad)
3. Compare R² - should be 0 to 1 for decent models

---

## Next Steps

1. ✅ Kill stuck Python processes (done)
2. ✅ Update pipeline with fixes (done)
3. 🚀 Run v2 pipeline with `--skip-catboost`
4. 📊 Review all plots and metrics
5. 🏆 Select best model for Pi deployment
