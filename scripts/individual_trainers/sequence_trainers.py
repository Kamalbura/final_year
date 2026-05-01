from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import optuna
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS, TARGET_COLUMN

class SequenceModel(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if model_type == "RNN":
            self.seq = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "LSTM":
            self.seq = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "GRU":
            self.seq = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == "BiLSTM":
            self.seq = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            return

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.seq(x)
        out = self.fc(out[:, -1, :])
        return out

class SequenceTrainer(AQBaseTrainer):
    def __init__(self, city, model_type, **kwargs):
        super().__init__(city, model_type, **kwargs)
        self.model_type = model_type

    def objective(self, trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"])), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_val"]), torch.from_numpy(data["y_val"])), batch_size=batch_size)
        
        model = SequenceModel(self.model_type, len(FEATURE_COLUMNS), hidden_dim, num_layers, 24, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(10): # Reduced epochs for tuning
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                preds = model(x_batch)
                val_loss += criterion(preds, y_batch).item()
        
        return val_loss / len(val_loader)

    def train(self, config: dict):
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"])), batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_test"]), torch.from_numpy(data["y_test"])), batch_size=config["batch_size"])
        
        model = SequenceModel(self.model_type, len(FEATURE_COLUMNS), config["hidden_dim"], config["num_layers"], 24, config["dropout"]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), self.output_dir / "best_model.pth")
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                preds = model(x_batch).cpu().numpy()
                all_preds.append(preds)
        
        preds = np.concatenate(all_preds, axis=0)
        y_scaler = data["y_scaler"]
        preds_inv = y_scaler.inverse_transform(preds)
        actuals_inv = y_scaler.inverse_transform(data["y_test"])
        
        metrics = {
            "rmse": float(np.sqrt(np.mean((preds_inv - actuals_inv)**2))),
            "mae": float(np.mean(np.abs(preds_inv - actuals_inv))),
            "r2": float(r2_score(actuals_inv, preds_inv))
        }
        
        self.save_results(metrics, config)
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", choices=["RNN", "LSTM", "GRU", "BiLSTM"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    trainer = SequenceTrainer(args.city, args.model)
    best_params = trainer.optimize(n_trials=args.trials)
    trainer.train(best_params)
