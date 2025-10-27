def prepare_training_data(df, target_col='is_winner'):
    """
    Prepare data for training
    
    Features to include:
    - Qualifying position (most important!)
    - Recent form (last 5 races avg position)
    - Championship standings position
    - Team performance metrics
    - Circuit-specific historical performance
    - Weather conditions
    """
    
    feature_cols = [
        # Qualifying
        'quali_position',
        'gap_to_pole',
        
        # Recent form
        'avg_finish_last_5',
        'avg_points_last_5',
        'podiums_last_5',
        'dnf_last_5',
        'win_rate_last_5',
        
        # Circuit-specific
        'avg_finish_at_circuit',
        'best_finish_at_circuit',
        'podium_rate_at_circuit',
        
        # Championship standings
        'championship_position',
        'points_total',
        'points_gap_to_leader',
        
        # Team performance
        'avg_team_position',
        'team_points_last_5',
        
        # Race characteristics
        'grid_position',
        'is_home_race',
        
        # Categorical encoded
        'driver_encoded',
        'team_encoded',
        'circuit_encoded'
    ]
    
    # Filter to only include available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df[target_col]
    
    return X, y, available_features


def temporal_train_test_split(df, test_year=2024):
    """
    Split data temporally to avoid data leakage
    Train on all years before test_year, test on test_year
    """
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] == test_year].copy()
    
    return train_df, test_df