#!/usr/bin/env python3
import sys
print(f"Python {sys.version}")

try:
    import sklearn
    print("✓ sklearn OK")
except ImportError as e:
    print(f"✗ sklearn MISSING: {e}")

try:
    import xgboost
    print("✓ xgboost OK")
except ImportError as e:
    print(f"✗ xgboost MISSING: {e}")

try:
    import lightgbm
    print("✓ lightgbm OK")
except ImportError as e:
    print(f"✗ lightgbm MISSING: {e}")

try:
    import catboost
    print("✓ catboost OK")
except ImportError as e:
    print(f"✗ catboost MISSING: {e}")

try:
    import joblib
    print("✓ joblib OK")
except ImportError as e:
    print(f"✗ joblib MISSING: {e}")

try:
    import pandas
    print("✓ pandas OK")
except ImportError as e:
    print(f"✗ pandas MISSING: {e}")

try:
    import numpy
    print("✓ numpy OK")
except ImportError as e:
    print(f"✗ numpy MISSING: {e}")
