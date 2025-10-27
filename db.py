"""
F1 Prediction - Complete End-to-End Pipeline
=============================================

This script connects FastF1 data collection to ML model training.
Run this to collect data, engineer features, and train models all in one go.
"""

import fastf1
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache
fastf1.Cache.enable_cache('/tmp/fastf1_cache')
from services.fastf1_data import collect_multiple_seasons
from core.features import engineer_all_features

# ============================================================================
# STEP 3: COMPLETE PIPELINE
# ============================================================================

def run_complete_pipeline(start_year=2022, end_year=2024, save_data=True):
    """
    Run the complete end-to-end pipeline:
    1. Collect data from FastF1
    2. Engineer features
    3. Save processed data
    """
    
    print("\n" + "="*70)
    print("F1 PREDICTION - COMPLETE DATA PIPELINE")
    print("="*70)
    print(f"Years: {start_year}-{end_year}")
    print("="*70)
    
    # Step 1: Collect raw data
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    
    raw_df = collect_multiple_seasons(start_year, end_year)
    
    if len(raw_df) == 0:
        print("\n✗ Failed to collect any data!")
        return None
    
    print(f"\n✓ Collected {len(raw_df)} driver-race records")
    print(f"  Races: {raw_df['race_id'].nunique()}")
    print(f"  Drivers: {raw_df['driver'].nunique()}")
    
    if save_data:
        raw_df.to_csv('/mnt/user-data/outputs/f1_raw_data.csv', index=False)
        print(f"  Saved: f1_raw_data.csv")
    
    # Step 2: Feature engineering
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    features_df = engineer_all_features(raw_df)
    
    if save_data:
        features_df.to_csv('f1_features.csv', index=False)
        print(f"\n✓ Saved: f1_features.csv")
    
    # Step 3: Show summary
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print(f"\nRaces by year:")
    print(features_df.groupby('year')['race_id'].nunique())
    
    print(f"\nWinner distribution (top 5 drivers):")
    print(features_df[features_df['is_winner']==1]['driver'].value_counts().head())
    
    print(f"\nPodium distribution (top 5 drivers):")
    print(features_df[features_df['is_podium']==1]['driver'].value_counts().head())
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Load the data: df = pd.read_csv('f1_features.csv')")
    print("2. Train models: See f1_train_models.py")
    
    return features_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    # This will collect 2022-2024 data (about 20-30 minutes)
    
    print("Starting F1 data collection and feature engineering...")
    print("\nThis will take approximately 20-30 minutes.")
    print("The script will collect data for 2022-2024 seasons.\n")
    
    df = run_complete_pipeline(start_year=2022, end_year=2024, save_data=True)
    
    if df is not None:
        print("\n" + "="*70)
        print("SUCCESS! Your data is ready for model training.")
        print("="*70)