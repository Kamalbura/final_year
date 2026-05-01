from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from scripts.individual_trainers.trainer_base import AQBaseTrainer, FEATURE_COLUMNS, TARGET_COLUMN

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [batch, num_nodes, features]
        # adj: [num_nodes, num_nodes]
        support = self.projection(x)
        output = torch.matmul(adj, support)
        return output

class STGCNModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        # Simplified ST-GCN: GCN for spatial + LSTM for temporal
        self.num_nodes = num_nodes
        self.gcn = GCNLayer(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [batch, seq_len, num_nodes, features]
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Spatial step (GCN on each time step)
        x = x.view(batch_size * seq_len, num_nodes, features)
        x = F.relu(self.gcn(x, adj))
        x = self.dropout(x)
        
        # Temporal step (LSTM on the node of interest - assuming node 0 is the target city)
        x = x.view(batch_size, seq_len, num_nodes, -1)
        # For simplicity, we model the sequence for the primary node
        target_node_seq = x[:, :, 0, :] 
        out, _ = self.lstm(target_node_seq)
        
        out = self.fc(out[:, -1, :])
        return out

class STGCNTrainer(AQBaseTrainer):
    def __init__(self, city, **kwargs):
        super().__init__(city, "STGCN", **kwargs)

    def _get_adj_matrix(self, num_nodes):
        # In a real scenario, this would be distance-based.
        # Here we use a learned or identity-based matrix for the single-city node.
        # To make it 'Spatio-Temporal', we'd need multiple cities' data.
        return torch.eye(num_nodes).to(self.device)

    def objective(self, trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        # Reshape for ST-GCN: [batch, seq_len, 1_node, features]
        X_train = data["X_train"][:, :, np.newaxis, :]
        X_val = data["X_val"][:, :, np.newaxis, :]
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(data["y_train"])), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(data["y_val"])), batch_size=batch_size)
        
        adj = self._get_adj_matrix(1)
        model = STGCNModel(1, len(FEATURE_COLUMNS), hidden_dim, 24).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(10):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(x_batch, adj)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                preds = model(x_batch, adj)
                val_loss += criterion(preds, y_batch).item()
        
        return val_loss / len(val_loader)

    def train(self, config: dict):
        df = self.load_data()
        data = self.prepare_data(df, lookback=168, horizon=24)
        
        X_train = data["X_train"][:, :, np.newaxis, :]
        X_test = data["X_test"][:, :, np.newaxis, :]
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(data["y_train"])), batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(data["y_test"])), batch_size=config["batch_size"])
        
        adj = self._get_adj_matrix(1)
        model = STGCNModel(1, len(FEATURE_COLUMNS), config["hidden_dim"], 24).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(x_batch, adj)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), self.output_dir / "best_model.pth")
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                preds = model(x_batch, adj).cpu().numpy()
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
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    trainer = STGCNTrainer(args.city)
    best_params = trainer.optimize(n_trials=args.trials)
    trainer.train(best_params)
