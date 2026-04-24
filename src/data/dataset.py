import os
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

CONFIG_PATH = os.path.join(os.getcwd(), 'config.yaml')

SCALER_MAP = {
    'standard': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler
}

def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def chronological_split(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio = cfg['data']['train_ratio']
    val_ratio = cfg['data']['val_ratio']
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

def fit_scaler(train: pd.DataFrame, scaler_name: str) -> Optional[object]:
    if scaler_name == 'none':
        return None
    scaler_cls = SCALER_MAP.get(scaler_name)
    if scaler_cls is None:
        raise ValueError(f"Unsupported scaler: {scaler_name}")
    scaler = scaler_cls()
    scaler.fit(train.values)
    return scaler

def apply_scaler(df: pd.DataFrame, scaler) -> pd.DataFrame:
    if scaler is None:
        return df
    arr = scaler.transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def window_arrays(data: pd.DataFrame, input_window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window input/target arrays.
    Returns X shape (samples, input_window, features) and Y shape (samples, horizon, features)."""
    values = data.values
    n_total = len(values)
    X_list = []
    Y_list = []
    end_input_limit = n_total - input_window - horizon + 1
    if end_input_limit <= 0:
        raise ValueError(
            f"Not enough rows ({n_total}) for input_window={input_window} and horizon={horizon}."
        )
    for start in range(end_input_limit):
        in_start = start
        in_end = start + input_window
        out_end = in_end + horizon
        X_list.append(values[in_start:in_end])
        Y_list.append(values[in_end:out_end])
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y

def build_datasets(cfg_override: Dict = None) -> Dict:
    cfg = load_config() if cfg_override is None else cfg_override
    cleaned_path = cfg['data']['cleaned_csv']
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_path}. Run preprocessing first.")

    df = pd.read_csv(cleaned_path, parse_dates=[cfg['data']['datetime_column']])
    df = df.set_index(cfg['data']['datetime_column']).sort_index()

    sensors = [c for c in cfg['data']['sensor_columns'] if c in df.columns]
    df = df[sensors].dropna()  # drop any rows still containing NaN after cleaning

    if len(df) < cfg['data']['input_window'] + cfg['data']['forecast_horizon']:
        raise ValueError(
            "Cleaned dataset is too short for the configured input window and forecast horizon."
        )

    train_df, val_df, test_df = chronological_split(df, cfg)

    scaler = fit_scaler(train_df, cfg['data']['scaler'])
    train_s = apply_scaler(train_df, scaler)
    val_s = apply_scaler(val_df, scaler)
    test_s = apply_scaler(test_df, scaler)

    input_w = cfg['data']['input_window']
    horizon = cfg['data']['forecast_horizon']

    X_train, Y_train = window_arrays(train_s, input_w, horizon)
    X_val, Y_val = window_arrays(val_s, input_w, horizon)
    X_test, Y_test = window_arrays(test_s, input_w, horizon)

    return {
        'scaler': scaler,
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
        'train_index': train_s.index,
        'val_index': val_s.index,
        'test_index': test_s.index,
        'features': sensors
    }

if __name__ == '__main__':
    datasets = build_datasets()
    for split in ['train', 'val', 'test']:
        X, Y = datasets[split]
        print(f"{split}: X={X.shape}, Y={Y.shape}")
    if datasets['scaler'] is not None:
        print(f"Scaler used: {datasets['scaler'].__class__.__name__}")
