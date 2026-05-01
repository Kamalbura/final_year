import torch
import sklearn
try:
    import onnxruntime
    print(f"ONNX Runtime: {onnxruntime.__version__}")
except:
    print("ONNX Runtime: Not installed")
print(f"PyTorch: {torch.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
