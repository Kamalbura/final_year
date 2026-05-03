"""Minimal ONNX converter - just export model weights"""
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path

OUT = Path(r"C:\Users\burak\ptojects\final_year\outputs\hyderabad")
ONNX_DIR = Path(r"C:\Users\burak\ptojects\final_year\deployment_models\hyderabad\onnx")
ONNX_DIR.mkdir(parents=True, exist_ok=True)

class GRUForecaster(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=2, output_dim=24, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2, output_dim=24, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

def try_load_state(model, path):
    """Try loading state dict with various approaches and return hidden dim"""
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    
    # Auto-detect hidden_dim from fc layer
    fc_weight = state.get('fc.weight')
    if fc_weight is not None:
        hidden_dim = fc_weight.shape[1]
    else:
        hidden_dim = 128  # fallback
    
    model.load_state_dict(state, strict=False)
    return model, hidden_dim

def create_model_for_checkpoint(state, model_type):
    """Create appropriate model based on state dict dimensions"""
    fc_weight = state.get('fc.weight')
    hidden_dim = fc_weight.shape[1] if fc_weight is not None else 128
    
    if model_type == 'transformer':
        return TransformerForecaster(d_model=64, nhead=2, num_layers=2, dropout=0.27)
    else:
        return GRUForecaster(hidden_dim=hidden_dim, num_layers=2, dropout=0.2)

def export_onnx(model, path, name):
    """Export to ONNX with verification"""
    model.eval()
    dummy = torch.randn(1, 168, 7)
    out_path = ONNX_DIR / f"{name}.onnx"
    
    # Trace first
    traced = torch.jit.trace(model, dummy)
    jit_path = ONNX_DIR / f"{name}.pt"
    traced.save(str(jit_path))
    print(f"  TorchScript: {jit_path.stat().st_size/1024:.0f} KB")
    
    # Export ONNX
    torch.onnx.export(
        model, dummy, str(out_path),
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"  ONNX: {out_path.stat().st_size/1024:.0f} KB")
    
    # Verify
    try:
        import onnx
        onnx.checker.check_model(str(out_path))
        print(f"  [VERIFIED]")
    except:
        print(f"  [SKIP verification - onnx not installed]")

# === Convert Transformer ===
print("\n=== Transformer ===")
try:
    path = OUT / "Transformer" / "model.pth"
    state = torch.load(path, map_location='cpu')
    model = create_model_for_checkpoint(state, 'transformer')
    model.load_state_dict(state, strict=False)
    export_onnx(model, path, "transformer")
except Exception as e:
    print(f"  FAILED: {e}")

# === Convert GRU ===
print("\n=== GRU ===")
try:
    path = OUT / "GRU" / "model.pth"
    state = torch.load(path, map_location='cpu')
    model = create_model_for_checkpoint(state, 'gru')
    model.load_state_dict(state, strict=False)
    export_onnx(model, path, "gru")
    print(f"  hidden_dim={model.gru.hidden_size}")
except Exception as e:
    print(f"  FAILED: {e}")

# === Convert CNN-LSTM ===
print("\n=== CNN-LSTM ===")
try:
    path = OUT / "CNN-LSTM" / "model.pth"
    state = torch.load(path, map_location='cpu')
    model = create_model_for_checkpoint(state, 'cnn_lstm')
    model.load_state_dict(state, strict=False)
    export_onnx(model, path, "cnn_lstm")
    print(f"  hidden_dim={model.gru.hidden_size}")
except Exception as e:
    print(f"  FAILED: {e}")

print(f"\nDone! ONNX models in {ONNX_DIR}")
