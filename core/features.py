import pandas as pd
import numpy as np

def calculate_rolling_features(df, lookback=5):
    """
    Calculate rolling performance metrics for each driver
    """
    df = df.sort_values(['driver', 'date']).reset_index(drop=True)
    
    # Initialize new columns
    rolling_features = []
    
    for driver in df['driver'].unique():
        driver_df = df[df['driver'] == driver].copy()
        
        # Rolling average finishing position
        driver_df['avg_finish_last_5'] = (
            driver_df['position']
            .rolling(window=lookback, min_periods=1)
            .mean()
            .shift(1)  # Shift to avoid data leakage
        )
        
        # Rolling average points
        driver_df['avg_points_last_5'] = (
            driver_df['points']
            .rolling(window=lookback, min_periods=1)
            .mean()
            .shift(1)
        )
        
        # Podium count in last N races
        driver_df['podiums_last_5'] = (
            driver_df['is_podium']
            .rolling(window=lookback, min_periods=1)
            .sum()
            .shift(1)
        )
        
        # DNF count in last N races
        driver_df['dnf_last_5'] = (
            driver_df['is_dnf']
            .rolling(window=lookback, min_periods=1)
            .sum()
            .shift(1)
        )
        
        # Win rate
        driver_df['win_rate_last_5'] = (
            driver_df['is_winner']
            .rolling(window=lookback, min_periods=1)
            .mean()
            .shift(1)
        )
        
        # Cumulative points (championship standing proxy)
        driver_df['points_total'] = driver_df['points'].cumsum().shift(1)
        
        rolling_features.append(driver_df)
    
    return pd.concat(rolling_features, ignore_index=True)


def calculate_circuit_features(df):
    """
    Calculate circuit-specific performance for each driver
    """
    circuit_features = []
    
    for (driver, circuit), group in df.groupby(['driver', 'circuit']):
        group = group.sort_values('date').copy()
        
        # Average finish at this circuit (expanding window, shifted)
        group['avg_finish_at_circuit'] = (
            group['position']
            .expanding()
            .mean()
            .shift(1)
        )
        
        # Best finish at this circuit
        group['best_finish_at_circuit'] = (
            group['position']
            .expanding()
            .min()
            .shift(1)
        )
        
        # Podium rate at circuit
        group['podium_rate_at_circuit'] = (
            group['is_podium']
            .expanding()
            .mean()
            .shift(1)
        )
        
        circuit_features.append(group)
    
    return pd.concat(circuit_features, ignore_index=True)


def calculate_team_features(df, lookback=5):
    """
    Calculate team-based features
    """
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    
    team_features = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        
        # Average team position
        team_df['avg_team_position'] = (
            team_df['position']
            .rolling(window=lookback, min_periods=1)
            .mean()
            .shift(1)
        )
        
        # Team points
        team_df['team_points_last_5'] = (
            team_df['points']
            .rolling(window=lookback, min_periods=1)
            .sum()
            .shift(1)
        )
        
        team_features.append(team_df)
    
    return pd.concat(team_features, ignore_index=True)


def calculate_championship_features(df):
    """
    Calculate championship standings features
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # For each race, calculate standings before that race
    standings_features = []
    
    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id].copy()
        race_date = race_df['date'].iloc[0]
        
        # Get all races before this one
        previous_races = df[df['date'] < race_date]
        
        if len(previous_races) > 0:
            # Calculate championship standings
            standings = (
                previous_races
                .groupby('driver')['points']
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            standings['championship_position'] = range(1, len(standings) + 1)
            
            # Points gap to leader
            if len(standings) > 0:
                leader_points = standings['points'].iloc[0]
                standings['points_gap_to_leader'] = leader_points - standings['points']
            else:
                standings['points_gap_to_leader'] = 0
            
            # Merge standings into race data
            race_df = race_df.merge(
                standings[['driver', 'championship_position', 'points_gap_to_leader']],
                on='driver',
                how='left'
            )
        else:
            # First race of season
            race_df['championship_position'] = 10  # Neutral position
            race_df['points_gap_to_leader'] = 0
        
        standings_features.append(race_df)
    
    return pd.concat(standings_features, ignore_index=True)


def add_categorical_encodings(df):
    """
    Encode categorical variables
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Encode driver
    driver_encoder = LabelEncoder()
    df['driver_encoded'] = driver_encoder.fit_transform(df['driver'])
    
    # Encode team
    team_encoder = LabelEncoder()
    df['team_encoded'] = team_encoder.fit_transform(df['team'])
    
    # Encode circuit
    circuit_encoder = LabelEncoder()
    df['circuit_encoded'] = circuit_encoder.fit_transform(df['circuit'])
    
    return df


def engineer_all_features(df):
    """
    Apply all feature engineering steps
    """
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