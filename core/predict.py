import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd

class F1WinnerPredictor:
    """
    Model to predict race winner
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the winner prediction model
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'xgboost':
            # XGBoost with class weights to handle imbalance
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_val_scaled, y_val)]
                self.model.fit(X_train_scaled, y_train, 
                             eval_set=eval_set, 
                             verbose=False)
            else:
                self.model.fit(X_train_scaled, y_train)
                
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
        
        self.feature_names = X_train.columns.tolist()
        
    def predict(self, X):
        """
        Predict winner (returns probabilities)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_top_n(self, X, race_ids, n=3):
        """
        Predict top N drivers for each race
        Useful for podium prediction
        """
        probas = self.predict(X)
        
        predictions = pd.DataFrame({
            'race_id': race_ids,
            'probability': probas
        })
        
        top_n_per_race = predictions.groupby('race_id').apply(
            lambda x: x.nlargest(n, 'probability')
        ).reset_index(drop=True)
        
        return top_n_per_race
    
    def get_feature_importance(self):
        """
        Get feature importance
        """
        if self.model_type in ['xgboost', 'lightgbm']:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None


class F1PodiumPredictor:
    """
    Model to predict podium finishers (top 3)
    Uses a different approach than winner prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """
        Train podium prediction model
        Top 3 is still imbalanced but less extreme than winner
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Calculate class weight
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X):
        """
        Predict podium probability
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class F1FastestLapPredictor:
    """
    Model to predict fastest lap
    This is tricky because it often goes to drivers who pit late for fresh tires
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """
        Train fastest lap prediction
        Consider adding features like:
        - Gap to cars ahead/behind (safe to pit)
        - Championship position (incentive to go for it)
        - Tire compound and age
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X):
        """
        Predict fastest lap probability
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

