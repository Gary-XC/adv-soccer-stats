import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class StratifiedStackingEnsemble:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.lgbm_params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'n_estimators': 300, 'learning_rate': 0.05,
            'max_depth': 5, 'num_leaves': 31, 'random_state': self.random_state
        }
        
        # Define the Stratified Feature Spaces
        self.td_cols = [] # Top-Down (Cohesion, Tactics, Elo)
        self.bu_cols = [] # Bottom-Up (Talent Density, Player Form)
        
        self.model_td = None
        self.model_bu = None
        self.meta_learner = None

    def _identify_feature_spaces(self, X):
        """Dynamically separates columns based on the td_ and bu_ prefixes."""
        self.td_cols = [c for c in X.columns if c.startswith('td_')]
        self.bu_cols = [c for c in X.columns if c.startswith('bu_')]
        print(f"Top-Down Features ({len(self.td_cols)}): {self.td_cols}")
        print(f"Bottom-Up Features ({len(self.bu_cols)}): {self.bu_cols}")

    def generate_oof_predictions(self, X_train, y_train_classes):
        self._identify_feature_spaces(X_train)
        
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        oof_td = np.zeros((len(X_train), 3))
        oof_bu = np.zeros((len(X_train), 3))
        
        print(f"Starting {self.n_splits}-Fold Stratified OOF Generation...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_classes)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr = y_train_classes[train_idx]
            
            # Model A: Team Cohesion & Elo
            m_td = lgb.LGBMClassifier(**self.lgbm_params)
            m_td.fit(X_tr[self.td_cols], y_tr)
            oof_td[val_idx] = m_td.predict_proba(X_val[self.td_cols]) # type: ignore
            
            # Model B: Player Talent Density
            m_bu = lgb.LGBMClassifier(**self.lgbm_params)
            m_bu.fit(X_tr[self.bu_cols], y_tr)
            oof_bu[val_idx] = m_bu.predict_proba(X_val[self.bu_cols]) # type: ignore
            
        return oof_td, oof_bu

    def train_meta_learner(self, oof_td, oof_bu, y_classes):
        """
        The Meta-Learner acts as the mathematical compromise between 
        Team Cohesion (Model A) and Player Talent (Model B).
        """
        print("Training Stratified Meta-Learner...")
        meta_features = np.hstack([oof_td, oof_bu]) # Shape: (N, 6)
        
        self.meta_learner = LogisticRegression(
            solver='lbfgs', max_iter=1000, C=1.0, random_state=self.random_state
        )
        self.meta_learner.fit(meta_features, y_classes)
        
        oof_preds = self.meta_learner.predict_proba(meta_features)
        print(f"Stratified Meta-Learner OOF Log-Loss: {log_loss(y_classes, oof_preds):.4f}")

    def fit_final_models(self, X_train, y_classes):
        print("Fitting Final Stratified Base Models...")
        self.model_td = lgb.LGBMClassifier(**self.lgbm_params)
        self.model_td.fit(X_train[self.td_cols], y_classes)
        
        self.model_bu = lgb.LGBMClassifier(**self.lgbm_params)
        self.model_bu.fit(X_train[self.bu_cols], y_classes)

    def predict(self, X_new):
        """Routes new data through both stratified models and the meta-learner."""
        probs_td = self.model_td.predict_proba(X_new[self.td_cols]) # type: ignore
        probs_bu = self.model_bu.predict_proba(X_new[self.bu_cols]) # type: ignore
        
        meta_features = np.hstack([probs_td, probs_bu]) # type: ignore
        return self.meta_learner.predict_proba(meta_features) # type: ignore

    def save_artifacts(self, output_dir="models/stratified_pipeline"):
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.model_td, os.path.join(output_dir, "model_top_down.pkl"))
        joblib.dump(self.model_bu, os.path.join(output_dir, "model_bottom_up.pkl"))
        joblib.dump(self.meta_learner, os.path.join(output_dir, "meta_learner.pkl"))
        joblib.dump({'td_cols': self.td_cols, 'bu_cols': self.bu_cols}, os.path.join(output_dir, "feature_spaces.pkl"))
        
        print(f"Stratified Artifacts saved to {output_dir}/")