#!/usr/bin/env python3
"""
Pi Inference Benchmark Script
Tests all 17 models for latency, memory, and CPU usage
Run on Raspberry Pi 4
"""

import json
import time
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import torch
import torch.nn as nn

# Model configurations
MODELS = {
    'statistical': ['ARIMA', 'SARIMA', 'VAR'],
    'classical_ml': ['SVR', 'Random_Forest', 'XGBoost', 'LightGBM'],
    'dl_sequence': ['RNN', 'LSTM', 'GRU', 'BiLSTM'],
    'dl_hybrid': ['CNN-LSTM', 'CNN-GRU', 'BiLSTM_Attention'],
    'transformers': ['Transformer', 'Informer', 'Autoformer'],
    'spatiotemporal': ['ST-GCN']
}

def get_system_info():
    """Get Pi system information"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_available': psutil.virtual_memory().available / (1024**2),  # MB
        'memory_total': psutil.virtual_memory().total / (1024**2),  # MB
        'disk_free': psutil.disk_usage('/').free / (1024**3),  # GB
        'temperature': get_cpu_temperature(),
        'timestamp': datetime.now().isoformat()
    }

def get_cpu_temperature():
    """Get CPU temperature (Pi specific)"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
        return temp
    except:
        return None

def benchmark_model(model_name, model_path, model_type):
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Load model
        print(f"  Loading model...")
        load_start = time.time()
        
        if model_type == 'classical_ml':
            model_data = joblib.load(model_path / 'model.joblib')
            model = model_data['model']
        else:
            # DL models - need to load architecture + weights
            model = load_dl_model(model_name, model_path)
        
        load_time = time.time() - load_start
        results['cold_start_ms'] = load_time * 1000
        print(f"  ✓ Loaded in {load_time:.3f}s")
        
        # Get memory after loading
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)
        results['memory_mb'] = memory_mb
        print(f"  Memory: {memory_mb:.1f} MB")
        
        # Warm-up inferences (not timed)
        print(f"  Warm-up (10 iterations)...")
        dummy_input = create_dummy_input(model_type)
        for _ in range(10):
            _ = run_inference(model, dummy_input, model_type)
        
        # Timed benchmark (100 iterations)
        print(f"  Benchmarking (100 iterations)...")
        latencies = []
        cpu_usages = []
        
        for i in range(100):
            cpu_before = psutil.cpu_percent(interval=None)
            start = time.time()
            
            output = run_inference(model, dummy_input, model_type)
            
            end = time.time()
            cpu_after = psutil.cpu_percent(interval=None)
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            cpu_usages.append(cpu_after)
        
        # Calculate statistics
        results['inference_ms_avg'] = np.mean(latencies)
        results['inference_ms_min'] = np.min(latencies)
        results['inference_ms_max'] = np.max(latencies)
        results['inference_ms_p95'] = np.percentile(latencies, 95)
        results['inference_ms_std'] = np.std(latencies)
        results['cpu_percent_avg'] = np.mean(cpu_usages)
        
        print(f"  ✓ Avg latency: {results['inference_ms_avg']:.2f} ms")
        print(f"  ✓ P95 latency: {results['inference_ms_p95']:.2f} ms")
        print(f"  ✓ Avg CPU: {results['cpu_percent_avg']:.1f}%")
        
        results['status'] = 'success'
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results

def create_dummy_input(model_type):
    """Create dummy input for inference"""
    if model_type == 'classical_ml':
        # Flattened features (168 hours * 5 features)
        return np.random.randn(1, 840).astype(np.float32)
    else:
        # Sequence input (batch=1, seq=168, features=5)
        return torch.randn(1, 168, 5)

def run_inference(model, input_data, model_type):
    """Run inference based on model type"""
    if model_type == 'classical_ml':
        return model.predict(input_data)
    else:
        with torch.no_grad():
            return model(input_data)

def load_dl_model(model_name, model_path):
    """Load deep learning model architecture and weights"""
    # This is a placeholder - actual implementation would load the specific architecture
    # For now, return None - in real deployment, we'd load the actual model class
    checkpoint = torch.load(model_path / 'model.pth', map_location='cpu')
    return checkpoint  # Returns state dict, need model architecture

def main():
    print("="*70)
    print("Raspberry Pi 4 - Model Inference Benchmark")
    print("="*70)
    
    # System info
    print("\n[SYSTEM INFO]")
    sys_info = get_system_info()
    print(f"  CPU: {sys_info['cpu_percent']:.1f}%")
    print(f"  Memory: {sys_info['memory_available']:.0f} MB / {sys_info['memory_total']:.0f} MB")
    print(f"  Disk Free: {sys_info['disk_free']:.1f} GB")
    if sys_info['temperature']:
        print(f"  Temperature: {sys_info['temperature']:.1f}°C")
    
    base_path = Path('/home/bura/projects/final_year/deployment_models/hyderabad')
    
    all_results = {
        'system_info': sys_info,
        'models': {}
    }
    
    # Benchmark all models
    for category, models in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Category: {category.upper()}")
        print(f"{'='*70}")
        
        for model_name in models:
            model_path = base_path / category / model_name
            
            if not model_path.exists():
                print(f"\n⚠️  {model_name}: Directory not found, skipping")
                continue
            
            model_type = 'classical_ml' if category == 'classical_ml' else 'deep_learning'
            result = benchmark_model(model_name, model_path, model_type)
            all_results['models'][model_name] = result
    
    # Save results
    output_file = Path('/home/bura/projects/final_year/deployment_models/benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n[SUMMARY]")
    successful = [m for m, r in all_results['models'].items() if r.get('status') == 'success']
    failed = [m for m, r in all_results['models'].items() if r.get('status') == 'failed']
    
    print(f"  Successful: {len(successful)}/{len(all_results['models'])}")
    print(f"  Failed: {len(failed)}/{len(all_results['models'])}")
    
    if successful:
        print("\n  Top 3 Fastest Models:")
        sorted_models = sorted(
            [(m, r) for m, r in all_results['models'].items() if r.get('status') == 'success'],
            key=lambda x: x[1]['inference_ms_avg']
        )[:3]
        
        for i, (name, result) in enumerate(sorted_models, 1):
            print(f"    {i}. {name}: {result['inference_ms_avg']:.2f} ms")

if __name__ == '__main__':
    main()
