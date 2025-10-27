from core.eval import evaluate_winner_prediction, evaluate_podium_prediction
from core.predict import F1WinnerPredictor, F1PodiumPredictor, F1FastestLapPredictor
from sklearn.metrics import roc_auc_score
from core.data_prep import prepare_training_data, temporal_train_test_split


def train_all_models(df):
    """
    Complete training pipeline for all prediction tasks
    """
    
    print("=" * 70)
    print("F1 PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 70)
    
    # Split data temporally
    train_df, test_df = temporal_train_test_split(df, test_year=2024)
    
    print(f"\nTraining set: {len(train_df)} records ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"Test set: {len(test_df)} records (2024)")
    
    # 1. Train Winner Predictor
    print("\n" + "-" * 70)
    print("Training Winner Predictor...")
    print("-" * 70)
    
    X_train, y_train_winner, features = prepare_training_data(train_df, 'is_winner')
    X_test, y_test_winner, _ = prepare_training_data(test_df, 'is_winner')
    
    winner_model = F1WinnerPredictor(model_type='xgboost')
    winner_model.train(X_train, y_train_winner)
    
    # Evaluate
    y_pred_winner = winner_model.predict(X_test)
    evaluate_winner_prediction(y_test_winner, y_pred_winner, test_df['race_id'])
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance = winner_model.get_feature_importance()
    print(importance.head(10))
    
    # 2. Train Podium Predictor
    print("\n" + "-" * 70)
    print("Training Podium Predictor...")
    print("-" * 70)
    
    X_train, y_train_podium, _ = prepare_training_data(train_df, 'is_podium')
    X_test, y_test_podium, _ = prepare_training_data(test_df, 'is_podium')
    
    podium_model = F1PodiumPredictor()
    podium_model.train(X_train, y_train_podium)
    
    # Evaluate
    y_pred_podium = podium_model.predict(X_test)
    evaluate_podium_prediction(y_test_podium, y_pred_podium, test_df['race_id'])
    
    # 3. Train Fastest Lap Predictor
    print("\n" + "-" * 70)
    print("Training Fastest Lap Predictor...")
    print("-" * 70)
    
    X_train, y_train_fl, _ = prepare_training_data(train_df, 'is_fastest_lap')
    X_test, y_test_fl, _ = prepare_training_data(test_df, 'is_fastest_lap')
    
    fl_model = F1FastestLapPredictor()
    fl_model.train(X_train, y_train_fl)
    
    # Evaluate
    y_pred_fl = fl_model.predict(X_test)
    
    print(f"\nFastest Lap Prediction:")
    print(f"  ROC-AUC: {roc_auc_score(y_test_fl, y_pred_fl):.3f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return {
        'winner_model': winner_model,
        'podium_model': podium_model,
        'fastest_lap_model': fl_model
    }