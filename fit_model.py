"""
Fix Model Saving
=================

This script re-trains and saves models in a way that's easier to load.
Run this once, then you can use predict_race.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

print("="*70)
print("RE-TRAINING AND SAVING MODELS")
print("="*70)

# Load your features
print("\nLoading training data...")
try:
    df = pd.read_csv('./data/f1_features.csv')
    print(f"✓ Loaded {len(df)} records")
except:
    print("✗ f1_features.csv not found!")
    print("Make sure you have run the data collection pipeline.")
    exit(1)

# Prepare data
print("\nPreparing data...")

# Feature columns (same as in training)
feature_cols = [
    'quali_position', 'grid_position', 
    'avg_finish_last_5', 'avg_points_last_5', 
    'podiums_last_5', 'dnf_last_5', 'win_rate_last_5',
    'championship_position', 'points_total', 'points_gap_to_leader',
    'avg_finish_at_circuit', 'best_finish_at_circuit', 'podium_rate_at_circuit',
    'avg_team_position', 'team_points_last_5',
    'driver_encoded', 'team_encoded', 'circuit_encoded'
]

# Filter to available features
available_features = [col for col in feature_cols if col in df.columns]

X = df[available_features]
y_winner = df['is_winner']
y_podium = df['is_podium']

# Split data (temporal)
train_df = df[df['year'] < 2024]
test_df = df[df['year'] >= 2023]

X_train = train_df[available_features]
y_train_winner = train_df['is_winner']
y_train_podium = train_df['is_podium']

X_test = test_df[available_features]
y_test_winner = test_df['is_winner']
y_test_podium = test_df['is_podium']

print(f"✓ Training set: {len(X_train)} records")
print(f"✓ Test set: {len(X_test)} records")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Winner Model
print("\n" + "="*70)
print("Training Winner Model...")
print("="*70)

scale_pos_weight = (len(y_train_winner) - y_train_winner.sum()) / y_train_winner.sum()

winner_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

winner_model.fit(X_train_scaled, y_train_winner)
print("✓ Winner model trained")

# Evaluate
y_pred_winner = winner_model.predict_proba(X_test_scaled)[:, 1]
test_results = pd.DataFrame({
    'race_id': test_df['race_id'],
    'actual': y_test_winner,
    'predicted_proba': y_pred_winner
})

predicted_winners = test_results.groupby('race_id').apply(
    lambda x: x['predicted_proba'].idxmax()
)

correct = 0
for race_id in predicted_winners.index:
    race_results = test_results[test_results['race_id'] == race_id]
    predicted_idx = predicted_winners[race_id]
    if race_results.loc[predicted_idx, 'actual'] == 1:
        correct += 1

accuracy = correct / len(predicted_winners)
print(f"Winner Accuracy: {accuracy:.2%} ({correct}/{len(predicted_winners)})")

# Train Podium Model
print("\n" + "="*70)
print("Training Podium Model...")
print("="*70)

scale_pos_weight = (len(y_train_podium) - y_train_podium.sum()) / y_train_podium.sum()

podium_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

podium_model.fit(X_train_scaled, y_train_podium)
print("✓ Podium model trained")

# Train Fastest Lap Model
print("\n" + "="*70)
print("Training Fastest Lap Model...")
print("="*70)

y_train_fastest = train_df['is_fastest_lap']
y_test_fastest = test_df['is_fastest_lap']

scale_pos_weight = (len(y_train_fastest) - y_train_fastest.sum()) / y_train_fastest.sum()

fastest_lap_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

fastest_lap_model.fit(X_train_scaled, y_train_fastest)
print("✓ Fastest lap model trained")

# Evaluate
from sklearn.metrics import roc_auc_score
y_pred_fastest = fastest_lap_model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test_fastest, y_pred_fastest)
print(f"Fastest Lap ROC-AUC: {auc:.3f}")

# Save models with scaler
print("\n" + "="*70)
print("Saving Models...")
print("="*70)

model_package = {
    'winner_model': winner_model,
    'podium_model': podium_model,
    'fastest_lap_model': fastest_lap_model,
    'scaler': scaler,
    'feature_names': available_features
}

with open('./models/f1_models.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✓ Saved: f1_models.pkl")
print("\nThis file contains:")
print("  - Winner prediction model")
print("  - Podium prediction model")
print("  - Fastest lap prediction model")
print("  - Feature scaler")
print("  - Feature names")

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
print("\nYou can now use these models for prediction.")
print("Run: python predict_race_simple.py")