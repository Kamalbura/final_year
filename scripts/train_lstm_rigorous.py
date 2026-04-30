import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
MODEL_NAME = "LSTM"
CITY = "delhi"
DATA_FILE = f"data/kaggle_dataset/clean_{CITY}_aq_1y.csv"
PLOT_DIR = Path("data/plots/LSTM")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("data/models/LSTM")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3

# --- Architecture ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, output_dim=1):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon * output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last hidden state
        pred = self.fc(out)
        return pred.view(-1, self.horizon, self.output_dim)

# --- Utilities ---
def window_xy(x_arr, y_arr, lookback, horizon):
    x_list, y_list = [], []
    for i in range(len(x_arr) - lookback - horizon + 1):
        x_list.append(x_arr[i : i + lookback])
        y_list.append(y_arr[i + lookback : i + lookback + horizon])
    return np.array(x_list), np.array(y_list)

def plot_learning_curve(train_losses, val_losses, filepath):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{MODEL_NAME} Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def plot_predictions(actual, predicted, filepath):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual AQI', color='green', marker='o')
    plt.plot(predicted, label='Predicted AQI', color='blue', linestyle='--', marker='x')
    plt.title(f'{MODEL_NAME} 24-Hour Horizon Prediction vs Actual')
    plt.xlabel('Hours into Future')
    plt.ylabel('US AQI')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

# --- Main Execution ---
def main():
    print(f"--- Starting Rigorous Training for {MODEL_NAME} on {CITY.upper()} ---")
    df = pd.read_csv(DATA_FILE)
    features = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    
    # Train/Val/Test Split (70/15/15)
    n = len(df)
    train_df = df.iloc[:int(n*0.7)]
    val_df = df.iloc[int(n*0.7):int(n*0.85)]
    test_df = df.iloc[int(n*0.85):]
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_s = x_scaler.fit_transform(train_df[features])
    y_train_s = y_scaler.fit_transform(train_df[['us_aqi']]).flatten()
    
    X_val_s = x_scaler.transform(val_df[features])
    y_val_s = y_scaler.transform(val_df[['us_aqi']]).flatten()
    
    X_test_s = x_scaler.transform(test_df[features])
    y_test_s = y_scaler.transform(test_df[['us_aqi']]).flatten()
    
    X_train, y_train = window_xy(X_train_s, y_train_s, LOOKBACK, HORIZON)
    X_val, y_val = window_xy(X_val_s, y_val_s, LOOKBACK, HORIZON)
    X_test, y_test = window_xy(X_test_s, y_test_s, LOOKBACK, HORIZON)
    
    # Dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = LSTMForecaster(input_dim=len(features), hidden_dim=64, num_layers=2, horizon=HORIZON).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    start_time = time.time()
    
    # Training Loop with Early Stopping
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item()
                
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_DIR / f"best_{MODEL_NAME}.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    training_time = time.time() - start_time
    
    # Load best model for testing
    model.load_state_dict(torch.load(MODEL_DIR / f"best_{MODEL_NAME}.pth", weights_only=True))
    model.eval()
    
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_actuals.append(y_batch.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)
    
    # Inverse transform to get actual AQI values
    # Because we flattened during fit, we need to reshape for inverse transform
    all_preds_inv = y_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    all_actuals_inv = y_scaler.inverse_transform(all_actuals.reshape(-1, 1)).reshape(all_actuals.shape)
    
    rmse = np.sqrt(mean_squared_error(all_actuals_inv, all_preds_inv))
    mae = mean_absolute_error(all_actuals_inv, all_preds_inv)
    
    print(f"\n--- {MODEL_NAME} Results ---")
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    model_size_mb = os.path.getsize(MODEL_DIR / f"best_{MODEL_NAME}.pth") / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    # Save plots
    plot_learning_curve(train_losses, val_losses, PLOT_DIR / f"{MODEL_NAME}_learning_curve.png")
    
    # Plot a random prediction sample
    sample_idx = np.random.randint(0, len(all_preds_inv))
    plot_predictions(all_actuals_inv[sample_idx], all_preds_inv[sample_idx], PLOT_DIR / f"{MODEL_NAME}_prediction_sample.png")
    
    print(f"Saved plots to {PLOT_DIR}")

if __name__ == "__main__":
    main()
