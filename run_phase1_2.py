import pandas as pd
import numpy as np
import os
import sys
import io

FILE_NAME = 'feeds.csv'
print(f"--- [Phase 1] Loading data from '{FILE_NAME}' ---")

if not os.path.exists(FILE_NAME):
    print(f"Error: '{FILE_NAME}' not found in working directory: {os.getcwd()}")
    sys.exit(2)

# attempt to read excel first, then csv
last_err = None
try:
    df = pd.read_excel(FILE_NAME)
    print(f"Loaded with pd.read_excel. Shape: {df.shape}")
except Exception as e_excel:
    last_err = e_excel
    try:
        df = pd.read_csv(FILE_NAME)
        print(f"Loaded with pd.read_csv. Shape: {df.shape}")
    except Exception as e_csv:
        print('Failed to read file with both read_excel and read_csv')
        print('read_excel error:', repr(last_err))
        print('read_csv error:', repr(e_csv))
        raise

print('\n--- [Phase 2] Initial Data Inspection ---\n')

print('[INFO] DataFrame Info:')
buf = io.StringIO()
df.info(buf=buf)
info = buf.getvalue()
print(info)

print('[HEAD] First 10 Rows:')
print(df.head(10).to_string())

print('\n[DESCRIBE] Statistical Summary:')
# Use to_string to avoid truncated output
print(df.describe(include='all').to_string())

print('\n[NULL COUNT] Missing values per column:')
print(df.isnull().sum().to_string())
