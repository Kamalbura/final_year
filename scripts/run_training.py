#!/usr/bin/env python3
"""
Quick runner for unified training pipeline
Usage: python run_training.py [--models MODEL ...] [--skip-dl] [--skip-ml] [--skip-stat]
"""

import subprocess
import sys
from pathlib import Path

# Run the unified pipeline
if __name__ == "__main__":
    cmd = [sys.executable, "scripts/unified_training_pipeline.py"] + sys.argv[1:]
    
    print("="*70)
    print("Starting Unified Training Pipeline")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    sys.exit(result.returncode)
