import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/kaggle_dataset")
OUTPUT_DIR = Path("outputs/final_deploy/LSTM")
OUTPUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(city: str, epochs: int = 30):
    output_dir = OUTPUT_DIR / city
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*40}")
    print(f"LSTM - {city}")
    print(f"{'='*40}")
    
    df = pd.read_csv(DATA_DIR / f"clean_{city}_aq_1y.csv")
    df = df.dropna()
    
    FEATURES = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
    TARGET = "us_aqi"
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_x.fit_transform(df[FEATURES])
    y = scaler_y.fit_transform(df[[TARGET]]).flatten()
    
    lookback = 24
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    
    model = LSTMModel(input_size=len(FEATURES), hidden_size=32, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy().flatten()
    
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(np.mean((preds - y_test_actual)**2))
    mae = np.mean(np.abs(preds - y_test_actual))
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_actual, preds)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(train_losses, 'b-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{city} - Convergence')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test_actual, preds, alpha=0.4, s=10)
    axes[1].plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'{city} - Parity\nRMSE={rmse:.2f}, R2={r2:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150)
    print(f"Saved: {output_dir / 'results.png'}")
    
    torch.save(model.state_dict(), output_dir / 'model.pth')
    
    return {"city": city, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

cities = ["delhi", "hyderabad", "bengaluru"]
results = []

for city in cities:
    r = train_lstm(city)
    results.append(r)

print(f"\n{'='*40}")
print("SUMMARY - LSTM (Fixed)")
print(f"{'='*40}")
for r in results:
    print(f"{r['city']:12} RMSE: {r['rmse']:.2f}  R2: {r['r2']:.3f}")

pd.DataFrame(results).to_csv(OUTPUT_DIR / "summary.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'summary.csv'}")