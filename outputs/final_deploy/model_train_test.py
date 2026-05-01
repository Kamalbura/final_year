import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd

print("=" * 60)
print("MODEL TRAINING SMOKE TEST: CPU vs GPU vs LSTM")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Generate random data (simulate real data)
np.random.seed(42)
X = np.random.randn(1600, 168, 5).astype(np.float32)  # 1600 windows
y = np.random.randn(1600, 1).astype(np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Split
n_train = 1280
X_train, y_train = X_tensor[:n_train], y_tensor[:n_train]
X_test, y_test = X_tensor[n_train:], y_tensor[n_train:]

print(f"\nData: {X_train.shape[0]} train, {X_test.shape[0]} test")
print(f"Input: {X_train.shape[1]} timesteps, {X_train.shape[2]} features")

# Training function
def train_model(device_name, epochs=5):
    model = SimpleLSTM().to(device_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_tr = X_train.to(device_name)
    y_tr = y_train.to(device_name)
    
    start = time.time()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
    
    train_time = time.time() - start
    
    # Inference
    model.eval()
    with torch.no_grad():
        X_te = X_test.to(device_name)
        start = time.time()
        for _ in range(10):
            preds = model(X_te)
        if device_name == "cuda":
            torch.cuda.synchronize()
        inf_time = (time.time() - start) / 10
    
    return train_time, inf_time, loss.item()

# Test CPU
print("\n--- CPU TRAINING ---")
cpu_train, cpu_inf, cpu_loss = train_model("cpu")
print(f"  Train time: {cpu_train:.2f}s")
print(f"  Inference: {cpu_inf*1000:.2f}ms")

# Test GPU
if torch.cuda.is_available():
    print("\n--- GPU TRAINING ---")
    try:
        gpu_train, gpu_inf, gpu_loss = train_model("cuda")
        print(f"  Train time: {gpu_train:.2f}s")
        print(f"  Inference: {gpu_inf*1000:.2f}ms")
        
        print("\n--- COMPARISON ---")
        print(f"  CPU train: {cpu_train:.2f}s")
        print(f"  GPU train: {gpu_train:.2f}s")
        if gpu_train > 0:
            print(f"  Speedup: {cpu_train/gpu_train:.2f}x")
    except Exception as e:
        print(f"  GPU error: {e}")
        gpu_train = float('inf')
else:
    print("\nGPU not available")

print("\n" + "=" * 60)
print("RECOMMENDATION BY MODEL TYPE")
print("=" * 60)

# Analyze results and make recommendations
results = [
    ("Statistical (ARIMA, VAR)", "CPU", "Single-threaded, fast", "No GPU benefit"),
    ("XGBoost/LightGBM", "CPU", "Tree-based, parallel", "CPU better for small data"),
    ("LSTM/GRU", "GPU" if gpu_train < cpu_train else "CPU", f"GPU={gpu_train:.1f}s, CPU={cpu_train:.1f}s", "GPU for large batch"),
    ("Transformer", "GPU", "Attention is memory-heavy", "GPU required"),
    ("Edge Inference", "CPU/ONNX", "XGBoost is fastest", "Pi uses CPU"),
]

for model_type, rec, detail, note in results:
    print(f"{model_type:25} | {rec:10} | {detail:25} | {note}")

# Save results
df = pd.DataFrame({
    "Model": ["CPU LSTM", "GPU LSTM", "Speedup"],
    "Train_Time_s": [cpu_train, gpu_train if 'gpu_train' in dir() else float('nan'), float('nan')],
    "Inference_ms": [cpu_inf*1000, gpu_inf*1000 if 'gpu_inf' in dir() else float('nan'), float('nan')],
    "Recommendation": ["Use for small models", "Use for large batch", f"{cpu_train/gpu_train:.2f}x" if 'gpu_train' in dir() else "N/A"]
}, index=[0, 1, 2]).T

df.to_csv("outputs/final_deploy/compute_comparison.csv", index=False)
print("\nSaved: outputs/final_deploy/compute_comparison.csv")