from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def evaluate_winner_prediction(y_true, y_pred_proba, race_ids):
    """
    Evaluate winner prediction performance
    """
    results = pd.DataFrame({
        'race_id': race_ids,
        'actual': y_true,
        'predicted_proba': y_pred_proba
    })
    
    # Get predicted winner per race (highest probability)
    predicted_winners = results.groupby('race_id').apply(
        lambda x: x['predicted_proba'].idxmax()
    )
    
    actual_winners = results[results['actual'] == 1]['race_id'].values
    
    # Calculate accuracy
    correct_predictions = 0
    total_races = len(predicted_winners)
    
    for race_id in predicted_winners.index:
        race_results = results[results['race_id'] == race_id]
        predicted_idx = predicted_winners[race_id]
        
        if race_results.loc[predicted_idx, 'actual'] == 1:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_races
    
    print(f"\nWinner Prediction Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Correct: {correct_predictions}/{total_races}")
    
    # ROC-AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"  ROC-AUC: {auc:.3f}")
    
    return accuracy


def evaluate_podium_prediction(y_true, y_pred_proba, race_ids, top_n=3):
    """
    Evaluate podium prediction (top 3)
    """
    results = pd.DataFrame({
        'race_id': race_ids,
        'actual': y_true,
        'predicted_proba': y_pred_proba
    })
    
    # Get top 3 predictions per race
    top_3_predictions = results.groupby('race_id').apply(
        lambda x: x.nlargest(top_n, 'predicted_proba').index.tolist()
    )
    
    # Calculate how many podium finishers we correctly predicted
    correct_in_top3 = []
    
    for race_id in top_3_predictions.index:
        race_results = results[results['race_id'] == race_id]
        predicted_indices = top_3_predictions[race_id]
        
        # How many actual podium finishers are in our top 3 predictions?
        actual_podium = race_results[race_results['actual'] == 1].index.tolist()
        
        overlap = len(set(predicted_indices) & set(actual_podium))
        correct_in_top3.append(overlap)
    
    avg_correct = np.mean(correct_in_top3)
    
    print(f"\nPodium Prediction Results:")
    print(f"  Average correct in top 3: {avg_correct:.2f}/3")
    print(f"  Percentage: {avg_correct/3:.2%}")
    
    return avg_correct