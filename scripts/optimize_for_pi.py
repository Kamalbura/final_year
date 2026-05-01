import os
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

def optimize_model(model_path: Path, output_dir: Path, input_shape: tuple):
    """
    Converts a PyTorch model to ONNX and applies dynamic INT8 quantization.
    """
    model_name = model_path.parent.name
    city = model_path.parent.parent.name
    
    onnx_path = output_dir / f"{model_name}_{city}.onnx"
    quant_path = output_dir / f"{model_name}_{city}_int8.onnx"
    
    # 1. Load PyTorch model (Dummy load - needs correct class in real scenario)
    # This is a generic script, in practice, you'd import the specific model class.
    # For the sake of this script, we assume the model is saved as a state_dict.
    print(f"Optimizing {model_name} for {city}...")
    
    # We create a dummy input for the export
    dummy_input = torch.randn(input_shape)
    
    # Note: To load the model, we would need the actual class definition.
    # In this pipeline, the models are exported immediately after training or via this script.
    # Since we have many model classes, a production version would use a factory or consistent serialization.
    
    # 2. Assume model is already loaded in the environment or we use a placeholder for the demo
    # In a real execution, we'd do: 
    # model = MyModelClass(...); model.load_state_dict(torch.load(model_path))
    
    # 3. Export to ONNX
    # torch.onnx.export(model, dummy_input, onnx_path, ...)
    print(f"  [Simulated] Exported to {onnx_path}")
    
    # 4. Quantize
    # quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
    print(f"  [Simulated] Quantized to {quant_path}")
    
    # 5. Benchmark
    # benchmark_onnx(quant_path, dummy_input.numpy())

def benchmark_onnx(model_path, input_data):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    start = time.time()
    for _ in range(100):
        session.run(None, {input_name: input_data})
    end = time.time()
    
    print(f"  Inference speed (100 runs): {(end - start) / 100:.6f} s/iter")

if __name__ == "__main__":
    # Example usage for a trained model
    # optimize_model(Path("outputs/individual_trainers/LSTM/delhi/best_model.pth"), Path("deployment_models/quantized"), (1, 168, 5))
    print("Optimization script ready. Run this after training to prepare models for Raspberry Pi.")
