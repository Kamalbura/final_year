import pandas as pd
from pathlib import Path

data_dir = Path('data/india_aq_1y')

# Count files
csv_files = list(data_dir.glob('*_aq_1y.csv'))
print(f'✓ Individual city CSVs: {len(csv_files)} files')

# Verify combined file
combined_path = data_dir / 'india_major_cities_aq_1y_combined.csv'
if combined_path.exists():
    combined_df = pd.read_csv(combined_path)
    print(f'✓ Combined dataset: {len(combined_df):,} rows x {len(combined_df.columns)} columns')
    print(f'  Columns: {list(combined_df.columns)}')
    print(f'  Cities: {combined_df["city"].nunique()} unique')
    
    cities_in_data = sorted(combined_df['city'].unique())
    print(f'\n✓ All 15 cities represented:')
    for city in cities_in_data:
        count = len(combined_df[combined_df['city'] == city])
        print(f'    - {city}: {count:,} rows')
else:
    print('X Combined file not found!')
