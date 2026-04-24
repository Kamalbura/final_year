import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import yaml

CONFIG_PATH = os.path.join(os.getcwd(), 'config.yaml')

def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _winsorize(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    if series.isna().all():
        return series
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lo, hi)

def _iqr_cap(series: pd.Series, k: float = 1.5) -> pd.Series:
    if series.isna().all():
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower, upper)

def _threshold_cap(series: pd.Series, limit: float) -> pd.Series:
    return series.mask(series > limit, np.nan)

def clean_dataframe(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    dt_col = cfg['data']['datetime_column']
    sensors = cfg['data']['sensor_columns']

    # Ensure datetime and index
    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce', utc=True)
    df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()

    # Optional winsorization
    if cfg['preprocessing']['winsorize']['enabled']:
        lq = cfg['preprocessing']['winsorize']['lower_quantile']
        uq = cfg['preprocessing']['winsorize']['upper_quantile']
        for c in sensors:
            if c in df.columns:
                df[c] = _winsorize(df[c], lq, uq)

    # Outlier handling
    method = cfg['preprocessing']['outlier']['method']
    if method == 'iqr':
        k = cfg['preprocessing']['outlier']['iqr_k']
        for c in sensors:
            if c in df.columns:
                df[c] = _iqr_cap(df[c], k)
    elif method == 'threshold':
        th = cfg['preprocessing']['outlier']['thresholds']
        for c in sensors:
            if c in df.columns and c in th:
                df[c] = _threshold_cap(df[c], th[c])

    # Missing handling (minimal leakage strategy)
    max_gap = cfg['preprocessing']['missing']['max_gap_linear']
    fill_strategy = cfg['preprocessing']['missing']['fill_strategy']
    for c in sensors:
        if c not in df.columns:
            continue
        s = df[c]
        # Linear fill for small gaps
        is_na = s.isna()
        if is_na.any():
            # Identify gaps
            na_groups = []
            gap_start = None
            for idx, missing in zip(s.index, is_na):
                if missing and gap_start is None:
                    gap_start = idx
                elif not missing and gap_start is not None:
                    na_groups.append((gap_start, idx))
                    gap_start = None
            # finalize trailing gap
            if gap_start is not None:
                na_groups.append((gap_start, s.index[-1]))
            to_interp = []
            for start, end in na_groups:
                gap_len = s.loc[start:end].isna().sum()
                if gap_len <= max_gap:
                    to_interp.append((start, end))
            # Interpolate selected small gaps only
            if to_interp:
                s = s.interpolate(limit=max_gap, limit_direction='both')
        if fill_strategy == 'forward_then_backward':
            s = s.ffill().bfill()
        df[c] = s

    return df

def resample(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    sensors = [c for c in cfg['data']['sensor_columns'] if c in df.columns]
    freq = cfg['data']['frequency']
    return df[sensors].resample(freq).mean()

def run_preprocessing(config_override: Dict = None) -> str:
    cfg = load_config() if config_override is None else config_override
    raw_path = cfg['data']['raw_csv']
    cleaned_path = cfg['data']['cleaned_csv']

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # Attempt both Excel and CSV
    try:
        df_raw = pd.read_excel(raw_path)
    except Exception:
        df_raw = pd.read_csv(raw_path)

    df_clean = clean_dataframe(df_raw.copy(), cfg)
    df_15T = resample(df_clean, cfg)
    df_15T.to_csv(cleaned_path, index=True)
    return cleaned_path

if __name__ == '__main__':
    path = run_preprocessing()
    print(f"Saved cleaned dataset to {path}")
