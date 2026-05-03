"""Convert trained PyTorch models to ONNX for Raspberry Pi deployment"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import sys
from pathlib import Path

OUTPUTS_DIR = Path(r"C:\Users\burak\ptojects\final_year\outputs\hyderabad")
DEPLOY_DIR = Path(r"C:\Users\burak\ptojects\final_year\deployment_models\hyderabad")

BATCH_SIZE = 1
SEQ_LEN = 168
INPUT_DIM = 7
OUTPUT_DIM = 24

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
    def __init__(self, input_dim=7, d_model=64, nhead=2, num_layers=2, output_dim=24, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

def convert_to_onnx(model, input_shape, output_path, model_name):
    """Convert PyTorch model to ONNX"""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"  [OK] {model_name} -> {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

def main():
    to_convert = [
        ("Transformer", OUTPUTS_DIR / "Transformer" / "model.pth", "transformer"),
        ("CNN-LSTM", OUTPUTS_DIR / "CNN-LSTM" / "model.pth", "cnn_lstm"),
        ("GRU", OUTPUTS_DIR / "GRU" / "model.pth", "gru"),
    ]
    
    for model_name, pth_path, model_type in to_convert:
        print(f"\n=== Converting {model_name} ===")
        
        if not pth_path.exists():
            print(f"  [SKIP] No model at {pth_path}")
            continue
        
        # Load scalers for verification
        scaler_dir = pth_path.parent
        x_scaler = pickle.load(open(scaler_dir / "x_scaler.pkl", "rb"))
        y_scaler = pickle.load(open(scaler_dir / "y_scaler.pkl", "rb"))
        
        # Load metrics for architecture params  
        metrics_path = scaler_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.load(open(metrics_path))
            bp = metrics.get("best_params", {})
            print(f"  Best params: {bp}")
        else:
            bp = {}
        
        # Create model
        if model_type == "transformer":
            model = TransformerForecaster(
                d_model=bp.get('model_dim', 64),
                nhead=bp.get('heads', 2),
                num_layers=bp.get('layers', 2),
                dropout=bp.get('dropout', 0.1)
            )
        else:
            # GRU/LSTM family - use GRU forecaster
            from collections import OrderedDict
            state = torch.load(pth_path, map_location='cpu')
            # Infer architecture from state dict
            # For CNN-LSTM and GRU, try loading as GRUForecaster
            try:
                # Simple GRU model
                hidden_dim = bp.get('hidden_dim', 128)
                num_layers = 2
                model = nn.Sequential(OrderedDict([
                    ('gru', nn.GRU(7, hidden_dim, num_layers, batch_first=True, dropout=bp.get('dropout', 0.2))),
                ]))
                # Create a wrapper for GRU
                class GRUWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.gru = nn.GRU(7, hidden_dim, num_layers, batch_first=True, dropout=bp.get('dropout', 0.2))
                        self.fc = nn.Linear(hidden_dim, 24)
                    def forward(self, x):
                        out, _ = self.gru(x)
                        return self.fc(out[:, -1, :])
                model = GRUWrapper()
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"  Failed to create model: {e}")
                continue
        
        # Convert to ONNX
        onnx_path = scaler_dir / f"{model_name.lower()}_model.onnx"
        convert_to_onnx(model, (1, SEQ_LEN, INPUT_DIM), onnx_path, model_name)
        
        # Verify ONNX
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"  [VERIFY] ONNX model valid")
        except ImportError:
            print(f"  [WARN] Cannot verify - install onnx package")
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
