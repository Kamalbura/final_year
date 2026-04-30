from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if model_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, model_dim: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos = PositionalEncoding(model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, heads, dim_feedforward=model_dim*2, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(model_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.pos(self.proj(x)))
        return self.head(encoded[:, -1, :])

class TransformerTrainer(AQBaseTrainer):
    def __init__(self, city, **kwargs):
        super().__init__(city, "Transformer", **kwargs)

    def objective(self, trial: optuna.Trial) -> float:
        model_dim = trial.suggest_categorical("model_dim", [32, 64, 128])
        layers = trial.suggest_int("layers", 1, 3)
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(data["X_train"]), torch.tensor(data["y_train"])),
            batch_size=64, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(data["X_val"]), torch.tensor(data["y_val"])),
            batch_size=64, shuffle=False
        )
        
        model = TransformerForecaster(len(FEATURE_COLUMNS), 24, model_dim, layers, heads, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(5):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                val_loss += criterion(model(x_b), y_b).item()
        
        return val_loss / len(val_loader)

    def train(self, config: dict):
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(data["X_train"]), torch.tensor(data["y_train"])),
            batch_size=config.get("batch_size", 64), shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(data["X_test"]), torch.tensor(data["y_test"])),
            batch_size=config.get("batch_size", 64), shuffle=False
        )
        
        model = TransformerForecaster(
            len(FEATURE_COLUMNS), 
            24, 
            config["model_dim"], 
            config["layers"], 
            config["heads"],
            config.get("dropout", 0.1)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()
        
        for epoch in range(config.get("epochs", 20)):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), self.output_dir / "best_model.pth")
        
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(self.device)
                p = model(x_b).cpu().numpy()
                preds.append(p)
                actuals.append(y_b.numpy())
        
        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)
        
        # Save one sample of predictions vs actuals for plotting
        sample_idx = len(actuals) - 1
        np.save(self.output_dir / "sample_pred.npy", preds[sample_idx])
        np.save(self.output_dir / "sample_actual.npy", actuals[sample_idx])
        
        y_scaler = data["y_scaler"]
        preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        actuals_inv = y_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
        
        # Save actual inverse for the sample as well
        np.save(self.output_dir / "sample_pred_inv.npy", y_scaler.inverse_transform(preds[sample_idx].reshape(-1, 1)).reshape(-1))
        np.save(self.output_dir / "sample_actual_inv.npy", y_scaler.inverse_transform(actuals[sample_idx].reshape(-1, 1)).reshape(-1))
        
        metrics = {
            "rmse": float(np.sqrt(np.mean((preds_inv - actuals_inv)**2))),
            "mae": float(np.mean(np.abs(preds_inv - actuals_inv)))
        }
        
        self.save_results(metrics, config)
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    trainer = TransformerTrainer(args.city)
    best_params = trainer.optimize(n_trials=args.trials)
    best_params["epochs"] = 20
    trainer.train(best_params)
