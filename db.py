"""
F1 Data Collection - Resilient Version
========================================

This script handles FastF1 API failures better by:
- Saving progress after each season
- Allowing manual retry
- Providing fallback options

STANDALONE - No imports from other scripts needed
"""

import fastf1
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Enable cache
fastf1.Cache.enable_cache('/tmp/fastf1_cache')


# ============================================================================
# DATA EXTRACTION FUNCTIONS (copied from pipeline)
# ============================================================================

def extract_driver_race_data(race, quali, year, round_number):
    """
    Extract per-driver data from a race session
    Returns a DataFrame with one row per driver
    """
    race_results = race.results
    circuit_name = race.event['Location']
    race_name = race.event['EventName']
    race_date = race.event['EventDate']
    
    driver_data = []
    
    for idx, driver in race_results.iterrows():
        try:
            driver_dict = {
                # Race identifiers
                'year': year,
                'round': round_number,
                'race_name': race_name,
                'circuit': circuit_name,
                'date': race_date,
                'race_id': f"{year}_{round_number}",
                
                # Driver info
                'driver': driver.get('Abbreviation', 'UNK'),
                'driver_number': driver.get('DriverNumber', 0),
                'team': driver.get('TeamName', 'Unknown'),
                
                # Race results
                'position': driver.get('Position', 20) if pd.notna(driver.get('Position')) else 20,
                'grid_position': driver.get('GridPosition', 20) if pd.notna(driver.get('GridPosition')) else 20,
                'points': driver.get('Points', 0) if pd.notna(driver.get('Points')) else 0,
                'status': driver.get('Status', 'Unknown'),
                
                # Target variables
                'is_winner': 1 if driver.get('Position') == 1 else 0,
                'is_podium': 1 if driver.get('Position', 20) <= 3 else 0,
                'is_points': 1 if driver.get('Position', 20) <= 10 else 0,
                'is_dnf': 1 if 'DNF' in str(driver.get('Status', '')) or 'Retired' in str(driver.get('Status', '')) else 0,
            }
            
            # Try to add fastest lap time if column exists
            driver_dict['fastest_lap'] = None
            for lap_col in ['FastestLap', 'FastestLapTime', 'BestLapTime']:
                if lap_col in driver.index and pd.notna(driver.get(lap_col)):
                    driver_dict['fastest_lap'] = driver[lap_col]
                    break
            
            # Add qualifying data if available
            if quali is not None:
                try:
                    quali_results = quali.results
                    driver_quali = quali_results[quali_results['Abbreviation'] == driver.get('Abbreviation', '')]
                    
                    if len(driver_quali) > 0:
                        driver_quali = driver_quali.iloc[0]
                        driver_dict['quali_position'] = driver_quali.get('Position', 20) if pd.notna(driver_quali.get('Position')) else 20
                        
                        # Q times with safe extraction
                        for q_num in ['Q1', 'Q2', 'Q3']:
                            try:
                                q_time = driver_quali.get(q_num)
                                driver_dict[f'{q_num.lower()}_time'] = q_time.total_seconds() if pd.notna(q_time) and hasattr(q_time, 'total_seconds') else None
                            except:
                                driver_dict[f'{q_num.lower()}_time'] = None
                    else:
                        driver_dict['quali_position'] = 20
                        driver_dict['q1_time'] = None
                        driver_dict['q2_time'] = None
                        driver_dict['q3_time'] = None
                except Exception as e:
                    print(f"      Warning: Could not extract quali data for {driver.get('Abbreviation', 'Unknown')}: {e}")
                    driver_dict['quali_position'] = driver_dict['grid_position']
                    driver_dict['q1_time'] = None
                    driver_dict['q2_time'] = None
                    driver_dict['q3_time'] = None
            else:
                driver_dict['quali_position'] = driver_dict['grid_position']
                driver_dict['q1_time'] = None
                driver_dict['q2_time'] = None
                driver_dict['q3_time'] = None
            
            driver_data.append(driver_dict)
            
        except Exception as e:
            print(f"      Warning: Error processing driver {idx}: {e}")
            continue
    
    # Find fastest lap driver using pick_fastest()
    try:
        # Get all laps from the race
        laps = race.laps
        if laps is not None and len(laps) > 0:
            # Use pick_fastest to get the fastest lap
            fastest_lap = laps.pick_fastest()
            if fastest_lap is not None:
                fastest_driver = fastest_lap['Driver']
                
                for d in driver_data:
                    d['is_fastest_lap'] = 1 if d['driver'] == fastest_driver else 0
            else:
                for d in driver_data:
                    d['is_fastest_lap'] = 0
        else:
            for d in driver_data:
                d['is_fastest_lap'] = 0
    except Exception as e:
        print(f"      Warning: Could not determine fastest lap: {e}")
        for d in driver_data:
            d['is_fastest_lap'] = 0
    
    return pd.DataFrame(driver_data)


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS (simplified versions)
# ============================================================================

def calculate_rolling_features(df, lookback=5):
    """Calculate rolling performance metrics for each driver"""
    df = df.sort_values(['driver', 'date']).reset_index(drop=True)
    
    rolling_features = []
    
    for driver in df['driver'].unique():
        driver_df = df[df['driver'] == driver].copy()
        
        # Rolling average finishing position
        driver_df['avg_finish_last_5'] = (
            driver_df['position'].rolling(window=lookback, min_periods=1).mean().shift(1)
        )
        
        # Rolling average points
        driver_df['avg_points_last_5'] = (
            driver_df['points'].rolling(window=lookback, min_periods=1).mean().shift(1)
        )
        
        # Podium count in last N races
        driver_df['podiums_last_5'] = (
            driver_df['is_podium'].rolling(window=lookback, min_periods=1).sum().shift(1)
        )
        
        # DNF count in last N races
        driver_df['dnf_last_5'] = (
            driver_df['is_dnf'].rolling(window=lookback, min_periods=1).sum().shift(1)
        )
        
        # Win rate
        driver_df['win_rate_last_5'] = (
            driver_df['is_winner'].rolling(window=lookback, min_periods=1).mean().shift(1)
        )
        
        # Cumulative points
        driver_df['points_total'] = driver_df['points'].cumsum().shift(1)
        
        rolling_features.append(driver_df)
    
    return pd.concat(rolling_features, ignore_index=True)


def calculate_circuit_features(df):
    """Calculate circuit-specific performance"""
    circuit_features = []
    
    for (driver, circuit), group in df.groupby(['driver', 'circuit']):
        group = group.sort_values('date').copy()
        
        group['avg_finish_at_circuit'] = group['position'].expanding().mean().shift(1)
        group['best_finish_at_circuit'] = group['position'].expanding().min().shift(1)
        group['podium_rate_at_circuit'] = group['is_podium'].expanding().mean().shift(1)
        
        circuit_features.append(group)
    
    return pd.concat(circuit_features, ignore_index=True)


def calculate_team_features(df, lookback=5):
    """Calculate team-based features"""
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    
    team_features = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        
        team_df['avg_team_position'] = (
            team_df['position'].rolling(window=lookback, min_periods=1).mean().shift(1)
        )
        
        team_df['team_points_last_5'] = (
            team_df['points'].rolling(window=lookback, min_periods=1).sum().shift(1)
        )
        
        team_features.append(team_df)
    
    return pd.concat(team_features, ignore_index=True)


def calculate_championship_features(df):
    """Calculate championship standings features"""
    df = df.sort_values('date').reset_index(drop=True)
    
    standings_features = []
    
    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id].copy()
        race_date = race_df['date'].iloc[0]
        
        previous_races = df[df['date'] < race_date]
        
        if len(previous_races) > 0:
            standings = (
                previous_races.groupby('driver')['points']
                .sum().sort_values(ascending=False).reset_index()
            )
            standings['championship_position'] = range(1, len(standings) + 1)
            
            if len(standings) > 0:
                leader_points = standings['points'].iloc[0]
                standings['points_gap_to_leader'] = leader_points - standings['points']
            else:
                standings['points_gap_to_leader'] = 0
            
            race_df = race_df.merge(
                standings[['driver', 'championship_position', 'points_gap_to_leader']],
                on='driver', how='left'
            )
        else:
            race_df['championship_position'] = 10
            race_df['points_gap_to_leader'] = 0
        
        standings_features.append(race_df)
    
    return pd.concat(standings_features, ignore_index=True)


def add_categorical_encodings(df):
    """Encode categorical variables"""
    driver_encoder = LabelEncoder()
    df['driver_encoded'] = driver_encoder.fit_transform(df['driver'])
    
    team_encoder = LabelEncoder()
    df['team_encoded'] = team_encoder.fit_transform(df['team'])
    
    circuit_encoder = LabelEncoder()
    df['circuit_encoded'] = circuit_encoder.fit_transform(df['circuit'])
    
    return df


def engineer_all_features(df):
    """Apply all feature engineering steps"""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    print("\n1. Calculating rolling features...")
    df = calculate_rolling_features(df, lookback=5)
    
    print("2. Calculating circuit-specific features...")
    df = calculate_circuit_features(df)
    
    print("3. Calculating team features...")
    df = calculate_team_features(df, lookback=5)
    
    print("4. Calculating championship features...")
    df = calculate_championship_features(df)
    
    print("5. Encoding categorical variables...")
    df = add_categorical_encodings(df)
    
    # Fill any remaining NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    print("\n✓ Feature engineering complete!")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Total records: {len(df)}")
    
    return df


# ============================================================================
# RESILIENT COLLECTION FUNCTIONS
# ============================================================================

def wait_and_retry(func, max_retries=5, initial_delay=5):
    """Execute function with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            result = func()
            return result
        except Exception as e:
            delay = initial_delay * (2 ** attempt)
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"    Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"    All {max_retries} attempts failed")
                raise e


def collect_season_resilient(year):
    """Collect a season with better error handling"""
    print(f"\n{'='*70}")
    print(f"Collecting {year} Season")
    print(f"{'='*70}")
    
    # Check if already cached
    cache_file = f'/tmp/f1_season_{year}.csv'
    if os.path.exists(cache_file):
        print(f"✓ Found cached data for {year}")
        try:
            df = pd.read_csv(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✓ Loaded {len(df)} records from cache")
            return df
        except:
            print("  Cache corrupted, re-collecting...")
    
    # Try to get schedule with retry
    print(f"\nFetching {year} schedule...")
    
    def get_schedule():
        return fastf1.get_event_schedule(year)
    
    try:
        schedule = wait_and_retry(get_schedule, max_retries=5, initial_delay=10)
        print(f"✓ Schedule loaded: {len(schedule)} events")
    except Exception as e:
        print(f"\n✗ Failed to load {year} schedule after multiple retries")
        print(f"Error: {e}")
        print(f"\nPossible solutions:")
        print(f"1. Wait a few minutes and try again (API might be rate limited)")
        print(f"2. Try a different year")
        print(f"3. Check your internet connection")
        return pd.DataFrame()
    
    # Collect each race
    all_data = []
    failed_rounds = []
    
    for idx, event in schedule.iterrows():
        if event.get('EventFormat') == 'testing':
            continue
        
        round_num = event['RoundNumber']
        event_name = event['EventName']
        
        print(f"\n  Round {round_num}: {event_name}")
        
        try:
            def load_race():
                race = fastf1.get_session(year, round_num, 'R')
                race.load()
                return race
            
            def load_quali():
                quali = fastf1.get_session(year, round_num, 'Q')
                quali.load()
                return quali
            
            # Load race with retry
            race = wait_and_retry(load_race, max_retries=3, initial_delay=5)
            print(f"    ✓ Race loaded")
            
            # Load qualifying with retry
            try:
                quali = wait_and_retry(load_quali, max_retries=3, initial_delay=5)
                print(f"    ✓ Qualifying loaded")
            except:
                print(f"    ⚠ Qualifying failed, using grid positions")
                quali = None
            
            # Extract data
            driver_data = extract_driver_race_data(race, quali, year, round_num)
            
            all_data.append(driver_data)
            print(f"    ✓ Collected {len(driver_data)} driver records")
            
            # Brief pause to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            failed_rounds.append((round_num, event_name))
    
    # Summary
    print(f"\n{'='*70}")
    if all_data:
        season_df = pd.concat(all_data, ignore_index=True)
        print(f"✓ Season {year} Complete: {len(season_df)} records from {len(all_data)} races")
        
        if failed_rounds:
            print(f"\n⚠ Failed to collect {len(failed_rounds)} races:")
            for round_num, name in failed_rounds:
                print(f"  - Round {round_num}: {name}")
        
        # Save to cache
        season_df.to_csv(cache_file, index=False)
        print(f"✓ Saved to cache: {cache_file}")
        
        return season_df
    else:
        print(f"✗ No data collected for {year}")
        return pd.DataFrame()


def main():
    """Main collection workflow with user control"""
    print("="*70)
    print("F1 DATA COLLECTION - RESILIENT VERSION")
    print("="*70)
    print("\nThis script handles API failures better by:")
    print("- Using exponential backoff retry")
    print("- Saving progress after each season")
    print("- Continuing even if some races fail")
    print("="*70)
    
    # Collect seasons one by one
    all_seasons = []
    years = [2022, 2023, 2024, 2025]
    
    for year in years:
        print(f"\n\n{'#'*70}")
        print(f"# COLLECTING SEASON {year}")
        print(f"{'#'*70}")
        
        season_df = collect_season_resilient(year)
        
        if len(season_df) > 0:
            all_seasons.append(season_df)
            print(f"\n✓ Season {year} successful!")
        else:
            print(f"\n✗ Season {year} failed or returned no data")
            
            # Ask user if they want to continue
            print(f"\nOptions:")
            print(f"1. Continue to next season (data collected so far will be saved)")
            print(f"2. Retry {year} (wait a bit and try again)")
            print(f"3. Skip {year} and continue")
            
            choice = input("\nEnter 1, 2, or 3: ").strip()
            
            if choice == '2':
                print(f"\nWaiting 30 seconds before retry...")
                time.sleep(30)
                season_df = collect_season_resilient(year)
                if len(season_df) > 0:
                    all_seasons.append(season_df)
            elif choice == '3':
                continue
            else:
                print(f"Continuing to next season...")
    
    # Combine and save
    if all_seasons:
        print(f"\n\n{'='*70}")
        print("COMBINING ALL SEASONS")
        print(f"{'='*70}")
        
        combined = pd.concat(all_seasons, ignore_index=True)
        
        print(f"\nTotal data collected:")
        print(f"  Total records: {len(combined)}")
        print(f"  Years: {sorted(combined['year'].unique())}")
        print(f"  Races: {combined['race_id'].nunique()}")
        print(f"  Drivers: {combined['driver'].nunique()}")
        
        # Save raw data
        combined.to_csv('./data/f1_raw_data.csv', index=False)
        print(f"\n✓ Saved: f1_raw_data.csv")
        
        # Now do feature engineering
        print(f"\n{'='*70}")
        print("FEATURE ENGINEERING")
        print(f"{'='*70}")
        
        try:
            features_df = engineer_all_features(combined)
            features_df.to_csv('./data/f1_features.csv', index=False)
            print(f"\n✓ Saved: f1_features.csv")
            
            print(f"\n{'='*70}")
            print("SUCCESS! Data ready for model training")
            print(f"{'='*70}")
            print("\nNext step:")
            print("  python f1_ml_models.py")
            
        except Exception as e:
            print(f"\n✗ Feature engineering failed: {e}")
            print(f"But raw data is saved, you can try feature engineering separately")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"\n✗ No data collected at all")
        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Wait 15-30 minutes (API might be rate limited)")
        print(f"3. Try running with just one year:")
        print(f"   season_df = collect_season_resilient(2022)")


if __name__ == "__main__":
    main()