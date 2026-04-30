from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from pathlib import Path
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMTrainer(AQBaseTrainer):
    def __init__(self, city, **kwargs):
        super().__init__(city, "LSTM", **kwargs)

    def objective(self, trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        num_layers = trial.suggest_int("num_layers", 1, 3)
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
        
        model = LSTMForecaster(len(FEATURE_COLUMNS), hidden_dim, num_layers, 24, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(5): # Small epochs for optimization
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
        data = self.prepare_data(df, lookback=config.get("lookback", 168), horizon=config.get("horizon", 24))
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(data["X_train"]), torch.tensor(data["y_train"])),
            batch_size=config.get("batch_size", 64), shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(data["X_test"]), torch.tensor(data["y_test"])),
            batch_size=config.get("batch_size", 64), shuffle=False
        )
        
        model = LSTMForecaster(
            len(FEATURE_COLUMNS), 
            config["hidden_dim"], 
            config["num_layers"], 
            config.get("horizon", 24), 
            config.get("dropout", 0.2)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        for epoch in range(config.get("epochs", 50)):
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
        
        y_scaler = data["y_scaler"]
        preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        actuals_inv = y_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
        
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
    
    trainer = LSTMTrainer(args.city)
    best_params = trainer.optimize(n_trials=args.trials)
    best_params["epochs"] = 20
    trainer.train(best_params)
