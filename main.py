"""
F1 Prediction Model - Machine Learning Implementation
======================================================

This script demonstrates how to build ML models for predicting:
1. Race winner
2. Podium finishers
3. Fastest lap
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from core.train import train_all_models


if __name__ == "__main__":
    import os
    
    # Check if features file exists
    features_file = 'f1_features.csv'
    
    if os.path.exists(features_file):
        print("Loading F1 features data...")
        df = pd.read_csv(features_file)
        
        print(f"\nDataset loaded:")
        print(f"  Total records: {len(df)}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        print(f"  Races: {df['race_id'].nunique()}")
        print(f"  Drivers: {df['driver'].nunique()}")
        
        print("\nStarting model training...")
        models = train_all_models(df)
        
        print("\n" + "="*70)
        print("MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        
    else:
        print("="*70)
        print("ERROR: Features file not found!")
        print("="*70)
        print("\nYou need to run the data collection pipeline first:")
        print("  python f1_complete_pipeline.py")
        print("\nThis will:")
        print("  1. Collect data from FastF1")
        print("  2. Engineer features")
        print("  3. Save f1_features.csv")
        print("\nThen you can run this script to train models.")
        print("="*70)