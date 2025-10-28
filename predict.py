"""
Simple F1 Race Predictor
=========================

Easy-to-use script for predicting F1 races.
Just update the qualifying results and run!
"""

import pandas as pd
import pickle

print("="*70)
print("F1 RACE PREDICTOR")
print("="*70)

# ============================================================================
# STEP 1: Load Models
# ============================================================================

print("\nLoading models...")

try:
    with open('./models/f1_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    winner_model = models['winner_model']
    podium_model = models['podium_model']
    fastest_lap_model = models['fastest_lap_model']
    scaler = models['scaler']
    feature_names = models['feature_names']
    
    print("✓ Models loaded successfully!")
    print("  - Winner prediction")
    print("  - Podium prediction")
    print("  - Fastest lap prediction")
    
except FileNotFoundError:
    print("\n✗ Models not found!")
    print("\nPlease run this first:")
    print("  python retrain_and_save.py")
    print("\nThis will create f1_models.pkl with all trained models.")
    exit(1)

# ============================================================================
# STEP 2: Load Historical Data
# ============================================================================

print("\nLoading historical data...")

try:
    historical_data = pd.read_csv('./data/f1_features.csv')
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    print(f"✓ Loaded {len(historical_data)} historical records")
except:
    print("✗ f1_features.csv not found!")
    exit(1)

# ============================================================================
# STEP 3: Enter Qualifying Results
# ============================================================================

print("\n" + "="*70)
print("ENTER QUALIFYING RESULTS")
print("="*70)

upcoming_race = pd.DataFrame({
    'driver': ['NOR','LEC','HAM','RUS','VER', 
               'ANT','SAI','PIA','HAD','BEA',
               'TSU','OCO','HUL','ALO','LAW',
               'BOR','ALB','GAS','STR','COL'],
    
    'team': ['McLaren','Ferrari','Ferrari','Mercedes','Red Bull Racing',
             'Mercedes','Williams','McLaren','Racing Bulls','Haas F1 Team',
             'Red Bull Racing','Haas F1 Team','Kick Sauber','Aston Martin',
             'Racing Bulls','Kick Sauber','Williams','Alpine','Aston Martin','Alpine'],
    
    'quali_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    
    'grid_position': [1, 2, 3, 4, 5, 6, 12, 7, 8, 9,
                      10, 11, 13, 14, 15, 16, 17, 18, 19, 20],
})

# ============================================================================
# STEP 4: Prepare Features
# ============================================================================

print("\n" + "="*70)
print("PREPARING FEATURES")
print("="*70)

race_features = []

for idx, driver_info in upcoming_race.iterrows():
    driver = driver_info['driver']
    
    # Get driver history
    driver_history = historical_data[historical_data['driver'] == driver].sort_values('date')
    
    if len(driver_history) > 0:
        recent = driver_history.tail(5)
        
        features = {
            'quali_position': driver_info['quali_position'],
            'grid_position': driver_info['grid_position'],
            'avg_finish_last_5': recent['position'].mean(),
            'avg_points_last_5': recent['points'].mean(),
            'podiums_last_5': recent['is_podium'].sum(),
            'dnf_last_5': recent['is_dnf'].sum(),
            'win_rate_last_5': recent['is_winner'].mean(),
            'points_total': driver_history['points'].sum(),
            'championship_position': 0,
            'points_gap_to_leader': 0,
            'avg_finish_at_circuit': driver_history['position'].mean(),
            'best_finish_at_circuit': driver_history['position'].min(),
            'podium_rate_at_circuit': driver_history['is_podium'].mean(),
            'avg_team_position': recent['position'].mean(),
            'team_points_last_5': recent['points'].sum(),
            'driver_encoded': driver_history['driver_encoded'].iloc[-1] if 'driver_encoded' in driver_history else 0,
            'team_encoded': driver_history['team_encoded'].iloc[-1] if 'team_encoded' in driver_history else 0,
            'circuit_encoded': driver_history['circuit_encoded'].iloc[-1] if 'circuit_encoded' in driver_history else 0,
        }
    else:
        # New driver defaults
        features = {
            'quali_position': driver_info['quali_position'],
            'grid_position': driver_info['grid_position'],
            'avg_finish_last_5': 10,
            'avg_points_last_5': 2,
            'podiums_last_5': 0,
            'dnf_last_5': 0,
            'win_rate_last_5': 0,
            'points_total': 0,
            'championship_position': 10,
            'points_gap_to_leader': 100,
            'avg_finish_at_circuit': 10,
            'best_finish_at_circuit': 10,
            'podium_rate_at_circuit': 0,
            'avg_team_position': 10,
            'team_points_last_5': 10,
            'driver_encoded': 0,
            'team_encoded': 0,
            'circuit_encoded': 0,
        }
    
    race_features.append({**{'driver': driver, 'team': driver_info['team']}, **features})

df_features = pd.DataFrame(race_features)

# Calculate championship positions
df_features = df_features.sort_values('points_total', ascending=False).reset_index(drop=True)
df_features['championship_position'] = range(1, len(df_features) + 1)
leader_points = df_features['points_total'].iloc[0] if len(df_features) > 0 else 0
df_features['points_gap_to_leader'] = leader_points - df_features['points_total']

df_features = df_features.fillna(0)

print("✓ Features prepared for all drivers")

# ============================================================================
# STEP 5: Make Predictions
# ============================================================================

print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)

# Extract feature columns
X = df_features[feature_names]

# Scale
X_scaled = scaler.transform(X)

# Predict
winner_probs = winner_model.predict_proba(X_scaled)[:, 1]
podium_probs = podium_model.predict_proba(X_scaled)[:, 1]
fastest_lap_probs = fastest_lap_model.predict_proba(X_scaled)[:, 1]

# Create results
results = pd.DataFrame({
    'driver': df_features['driver'],
    'team': df_features['team'],
    'quali_position': df_features['quali_position'],
    'win_probability': winner_probs,
    'podium_probability': podium_probs,
    'fastest_lap_probability': fastest_lap_probs
})

results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)

print("✓ Predictions complete!")

# ============================================================================
# STEP 6: Display Results
# ============================================================================

print("\n" + "="*70)
print("🏆 RACE PREDICTIONS")
print("="*70)

print("\n🥇 Winner Prediction (Top 5):")
print("-" * 70)
for idx, row in results.head(5).iterrows():
    print(f"{idx+1}. {row['driver']:3s} ({row['team'][:20]:20s}) - "
          f"{row['win_probability']*100:5.1f}% - Quali P{row['quali_position']:.0f}")

print("\n🏁 Podium Prediction:")
print("-" * 70)
podium = results.nlargest(3, 'podium_probability')
for idx, (_, row) in enumerate(podium.iterrows(), 1):
    print(f"{idx}. {row['driver']:3s} ({row['team'][:20]:20s}) - "
          f"{row['podium_probability']*100:5.1f}%")

print("\n⚡ Fastest Lap Prediction (Top 5):")
print("-" * 70)
fastest = results.nlargest(5, 'fastest_lap_probability')
for idx, (_, row) in enumerate(fastest.iterrows(), 1):
    print(f"{idx}. {row['driver']:3s} ({row['team'][:20]:20s}) - "
          f"{row['fastest_lap_probability']*100:5.1f}%")

print("\n📊 Full Grid Prediction:")
print("-" * 70)
print(f"{'Pos':<4} {'Driver':<7} {'Team':<25} {'Win %':<8} {'Podium %':<10} {'FL %':<8} {'Quali'}")
print("-" * 70)

for idx, row in results.iterrows():
    print(f"{idx+1:<4} {row['driver']:<7} {row['team'][:24]:<25} "
          f"{row['win_probability']*100:<8.1f} {row['podium_probability']*100:<10.1f} "
          f"{row['fastest_lap_probability']*100:<8.1f} "
          f"P{row['quali_position']:.0f}")

# Save predictions
results.to_csv('./output/race_predictions.csv', index=False)
print("\n✓ Saved: race_predictions.csv")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print("\nTo predict a real race:")
print("1. Wait for qualifying to finish")
print("2. Edit this script and update the 'upcoming_race' DataFrame")
print("3. Run: python predict_race_simple.py")