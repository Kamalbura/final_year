import json
import pandas as pd
from pathlib import Path

models_dir = Path('outputs/hyderabad')
results = []

model_order = [
    'ARIMA', 'SARIMA', 'VAR',
    'SVR', 'Random_Forest', 'XGBoost', 'LightGBM',
    'RNN', 'LSTM', 'GRU', 'BiLSTM',
    'CNN-LSTM', 'CNN-GRU', 'BiLSTM_Attention',
    'Transformer', 'Informer', 'Autoformer', 'ST-GCN'
]

phase_map = {
    'ARIMA': 'Statistical', 'SARIMA': 'Statistical', 'VAR': 'Statistical',
    'SVR': 'Classical ML', 'Random_Forest': 'Classical ML', 'XGBoost': 'Classical ML', 'LightGBM': 'Classical ML',
    'RNN': 'DL Sequence', 'LSTM': 'DL Sequence', 'GRU': 'DL Sequence', 'BiLSTM': 'DL Sequence',
    'CNN-LSTM': 'DL Hybrid', 'CNN-GRU': 'DL Hybrid', 'BiLSTM_Attention': 'DL Hybrid',
    'Transformer': 'Transformer', 'Informer': 'Transformer', 'Autoformer': 'Transformer',
    'ST-GCN': 'Spatio-Temporal'
}

for model_name in model_order:
    metrics_file = models_dir / model_name / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
        
        results.append({
            'Model': model_name,
            'Phase': phase_map.get(model_name, 'Unknown'),
            'RMSE': round(data.get('rmse', 0), 2),
            'MAE': round(data.get('mae', 0), 2),
            'R2': round(data.get('r2', 0), 3),
            'Train_Time_s': round(data.get('train_seconds', 0), 1),
            'Inference_Time_s': round(data.get('inference_seconds', 0), 3)
        })

df = pd.DataFrame(results)
df = df.sort_values('RMSE')

print('\n' + '='*90)
print('COMPREHENSIVE MODEL COMPARISON - Hyderabad Dataset')
print('='*90)
print(df.to_string(index=False))
print('\n' + '='*90)
print(f'Total Models: {len(df)}')
print(f'Best Model: {df.iloc[0]["Model"]} (RMSE: {df.iloc[0]["RMSE"]})')
print(f'Worst Model: {df.iloc[-1]["Model"]} (RMSE: {df.iloc[-1]["RMSE"]})')
print('='*90)

# Save to CSV
df.to_csv('outputs/hyderabad/all_models_comparison.csv', index=False)
print('\n✓ Saved to: outputs/hyderabad/all_models_comparison.csv')

# Also save summary
summary = f"""
MODEL TRAINING SUMMARY
=====================
Total Models Trained: {len(df)}
Best Overall Model: {df.iloc[0]['Model']} (RMSE: {df.iloc[0]['RMSE']}, R²: {df.iloc[0]['R2']})
Best Classical ML: {df[df['Phase']=='Classical ML'].iloc[0]['Model']} (RMSE: {df[df['Phase']=='Classical ML'].iloc[0]['RMSE']})
Best DL Model: {df[df['Phase'].str.contains('DL')].iloc[0]['Model']} (RMSE: {df[df['Phase'].str.contains('DL')].iloc[0]['RMSE']})
Best Transformer: {df[df['Phase']=='Transformer'].iloc[0]['Model']} (RMSE: {df[df['Phase']=='Transformer'].iloc[0]['RMSE']})

TOP 3 MODELS FOR DEPLOYMENT:
1. {df.iloc[0]['Model']} - RMSE: {df.iloc[0]['RMSE']}, R²: {df.iloc[0]['R2']}
2. {df.iloc[1]['Model']} - RMSE: {df.iloc[1]['RMSE']}, R²: {df.iloc[1]['R2']}
3. {df.iloc[2]['Model']} - RMSE: {df.iloc[2]['RMSE']}, R²: {df.iloc[2]['R2']}
"""

with open('outputs/hyderabad/training_summary.txt', 'w') as f:
    f.write(summary)

print(summary)
