import pandas as pd
import numpy as np

INPUT_FILE = 'biometric_data.csv'
CLEANED_FILE = 'set3_cleaned_new.csv'

try:
    df = pd.read_csv(INPUT_FILE, header=None)
    initial_count = len(df)
    
    # 1. REMOVE ONLY TRUE DEAD ROWS
    # We only remove rows where the average is 0 (meaning no data at all)
    raw_data_block = df.iloc[:, 1:51]
    df = df[raw_data_block.mean(axis=1) != 0]
    
    # 2. STANDARDIZE LABELS
    def fix_label(label):
        label = str(label).upper()
        if 'SILENT' in label: return 'SILENT'
        if 'HUM' in label or 'NORMAL' in label or '_S' in label: return 'NORMAL'
        if 'FIRM' in label or 'PRESS' in label: return 'FIRM'
        return 'NORMAL' # Default to Normal if it looks like touch data

    df[0] = df[0].apply(fix_label)
    
    # 3. CLEAN NUMERICS
    # Ensure the 6 features at the end are clean
    df.iloc[:, -6:] = df.iloc[:, -6:].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    df.to_csv(CLEANED_FILE, index=False, header=None)
    
    print(f"Success! Kept {len(df)} rows out of {initial_count}.")
    print(f"Data saved to {CLEANED_FILE}")

except Exception as e:
    print(f"Error: {e}")