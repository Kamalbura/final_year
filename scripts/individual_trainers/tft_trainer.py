from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS, TARGET_COLUMN

# Simplified Temporal Fusion Transformer (TFT) Components
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != output_dim:
            self.res = nn.Linear(input_dim, output_dim)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        out = self.norm(self.res(x) + g * h)
        return self.dropout(out)

class TFTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_proj(x)
        x = self.grn(x)
        
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.post_attn_grn(x + attn_out)
        
        # Predicting horizon from the last time step
        out = self.fc(x[:, -1, :])
        return out

class TFTTrainer(AQBaseTrainer):
    def __init__(self, city, **kwargs):
        super().__init__(city, "TFT", **kwargs)

    def objective(self, trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"])), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(data["X_val"]), torch.from_numpy(data["y_val"])), batch_size=batch_size)
        
        model = TFTModel(len(FEATURE_COLUMNS), hidden_dim, 24, num_heads, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(15):
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
        
        model = TFTModel(len(FEATURE_COLUMNS), config["hidden_dim"], 24, config["num_heads"], config["dropout"]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()
        
        for epoch in range(60):
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
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()
    
    trainer = TFTTrainer(args.city)
    best_params = trainer.optimize(n_trials=args.trials)
    trainer.train(best_params)
