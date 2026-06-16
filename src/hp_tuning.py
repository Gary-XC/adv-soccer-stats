import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import joblib
import os
import sys
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_splitter import DataSplitter

optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    print("=== Loading Data ===")
    match_df = pd.read_csv("data/hybrid_matrices.csv", parse_dates=['date'])
    splitter = DataSplitter(config_path="../config.yaml")
    X_train, y_train, train_df, X_test, y_test, test_df = splitter.split_data(match_df)

    # Force the test set to ONLY include FIFA World Cup matches (league_id == 1)
    wc_test_mask = test_df['league_id'] == 1
    X_test = X_test[wc_test_mask].copy()
    y_test = y_test[wc_test_mask].copy()
    test_df = test_df[wc_test_mask].copy()
    
    print(f"Test set strictly filtered to {len(X_test)} World Cup matches only.")
    
    # Identify feature spaces
    td_cols = [c for c in X_train.columns if c.startswith('td_')]
    bu_cols = [c for c in X_train.columns if c.startswith('bu_')]
    
    # Map targets to 0 (Home), 1 (Draw), 2 (Away)
    y_train_classes = np.ones(len(y_train), dtype=int)
    y_train_classes[y_train['goals_total_home'] > y_train['goals_total_away']] = 0
    y_train_classes[y_train['goals_total_home'] < y_train['goals_total_away']] = 2
    
    y_test_classes = np.ones(len(y_test), dtype=int)
    y_test_classes[y_test['goals_total_home'] > y_test['goals_total_away']] = 0
    y_test_classes[y_test['goals_total_home'] < y_test['goals_total_away']] = 2

    # Creating a validation split of the data for Optuna
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train_classes[:split_idx], y_train_classes[split_idx:]
    
    dtrain_td = lgb.Dataset(X_tr[td_cols], label=y_tr)
    dval_td = lgb.Dataset(X_val[td_cols], label=y_val, reference=dtrain_td)
    
    dtrain_bu = lgb.Dataset(X_tr[bu_cols], label=y_tr)
    dval_bu = lgb.Dataset(X_val[bu_cols], label=y_val, reference=dtrain_bu)

    # ==========================================
    # OPTUNA OBJECTIVES
    # ==========================================
    def objective_td(trial):
        param = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42,
            'feature_pre_filter': False,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
        }
        
        model = lgb.train(
            param, dtrain_td, valid_sets=[dval_td], num_boost_round=1000, 
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)]
        )
        preds = model.predict(X_val[td_cols])
        return log_loss(y_val, preds) # type: ignore

    def objective_bu(trial):
        param = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42,
            'feature_pre_filter': False, # CRITICAL FIX
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
        }
        
        model = lgb.train(
            param, dtrain_bu, valid_sets=[dval_bu], num_boost_round=1000, 
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(0)]
        )
        preds = model.predict(X_val[bu_cols])
        return log_loss(y_val, preds) # type: ignore

    # ==========================================
    # RUN OPTUNA STUDIES
    # ==========================================
    print("=== Starting Optuna Tuning for Top-Down Model (30 Trials) ===")
    study_td = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study_td.optimize(objective_td, n_trials=30)  # type: ignore
    
    best_params_td = study_td.best_params.copy()
    best_params_td.update({
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 
        'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42
    })
    print(f"🏆 Best TD Validation Log-Loss: {study_td.best_value:.4f}")

    print("=== Starting Optuna Tuning for Bottom-Up Model (30 Trials) ===")
    study_bu = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study_bu.optimize(objective_bu, n_trials=30) # type: ignore
    
    best_params_bu = study_bu.best_params.copy()
    best_params_bu.update({
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 
        'verbosity': -1, 'boosting_type': 'gbdt', 'seed': 42
    })
    print(f"🏆 Best BU Validation Log-Loss: {study_bu.best_value:.4f}")

    # ==========================================
    # TRAIN FINAL MODELS WITH BEST PARAMS
    # ==========================================
    print("=== Training Final Models with Optimized Hyperparameters ===")
    
    final_model_td = lgb.LGBMClassifier(**best_params_td, n_estimators=1000)
    final_model_td.fit(X_train[td_cols], y_train_classes)
    
    final_model_bu = lgb.LGBMClassifier(**best_params_bu, n_estimators=1000)
    final_model_bu.fit(X_train[bu_cols], y_train_classes)

    # ==========================================
    # GENERATE OOF FOR META-LEARNER
    # ==========================================
    print("=== Generating OOF for Meta-Learner ===")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_td = np.zeros((len(X_train), 3))
    oof_bu = np.zeros((len(X_train), 3))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train_classes)):
        m_td = lgb.LGBMClassifier(**best_params_td, n_estimators=1000)
        m_td.fit(X_train.iloc[tr_idx][td_cols], y_train_classes[tr_idx])
        oof_td[val_idx] = m_td.predict_proba(X_train.iloc[val_idx][td_cols]) # type: ignore
        
        m_bu = lgb.LGBMClassifier(**best_params_bu, n_estimators=1000)
        m_bu.fit(X_train.iloc[tr_idx][bu_cols], y_train_classes[tr_idx])
        oof_bu[val_idx] = m_bu.predict_proba(X_train.iloc[val_idx][bu_cols]) # type: ignore

    # ==========================================
    # TRAIN META-LEARNER & EVALUATE
    # ==========================================
    meta_features = np.hstack([oof_td, oof_bu])
    meta_learner = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, random_state=42)
    meta_learner.fit(meta_features, y_train_classes)
    
    # Evaluate on STRICT World Cup Test Set
    test_preds_td = final_model_td.predict_proba(X_test[td_cols])
    test_preds_bu = final_model_bu.predict_proba(X_test[bu_cols])
    test_meta_features = np.hstack([test_preds_td, test_preds_bu]) # type: ignore
    final_probs = meta_learner.predict_proba(test_meta_features)
    
    test_ll = log_loss(y_test_classes, final_probs)
    test_acc = np.mean(np.argmax(final_probs, axis=1) == y_test_classes)
    print(f"\n🎯 OPTIMIZED World Cup Test Set Log-Loss: {test_ll:.4f} | Accuracy: {test_acc:.4f}")

    # ==========================================
    # SAVE VERSIONED ARTIFACTS & UPDATE LATEST PATH
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    versioned_dir = f"../models/optimized_pipeline_{timestamp}"
    os.makedirs(versioned_dir, exist_ok=True)
    
    print(f"\n=== Saving Versioned Artifacts to: {versioned_dir} ===")
    joblib.dump(final_model_td, os.path.join(versioned_dir, "model_top_down.pkl"))
    joblib.dump(final_model_bu, os.path.join(versioned_dir, "model_bottom_up.pkl"))
    joblib.dump(meta_learner, os.path.join(versioned_dir, "meta_learner.pkl"))
    joblib.dump({'td_cols': td_cols, 'bu_cols': bu_cols}, os.path.join(versioned_dir, "feature_spaces.pkl"))
    
    # Update the latest model pointer
    with open("models/latest_model_path.txt", "w") as f:
        f.write(versioned_dir)
    print("✅ 'latest_model_path.txt' updated.")

    # ==========================================
    # SHAP EXPLAINABILITY
    # ==========================================
    print("=== Generating SHAP Explanations ===")
    explainer_td = shap.TreeExplainer(final_model_td)
    shap_values_td = explainer_td.shap_values(X_test[td_cols])
    
    explainer_bu = shap.TreeExplainer(final_model_bu)
    shap_values_bu = explainer_bu.shap_values(X_test[bu_cols])

    # Helper function to safely extract Class 0 (Home Win) SHAP values
    def get_class0_shap(shap_vals):
        if isinstance(shap_vals, list):
            return shap_vals[0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            return shap_vals[:, :, 0]
        else:
            return shap_vals

    shap_class0_td = get_class0_shap(shap_values_td)
    shap_class0_bu = get_class0_shap(shap_values_bu)

    # Plot and save SHAP summary plots
    plt.figure()
    shap.summary_plot(shap_class0_td, X_test[td_cols], show=False, plot_type="dot")
    plt.title("Top-Down Model SHAP (Home Win Drivers - World Cup)")
    plt.tight_layout()
    plt.savefig(os.path.join(versioned_dir, "shap_top_down.png"))
    plt.close()

    plt.figure()
    shap.summary_plot(shap_class0_bu, X_test[bu_cols], show=False, plot_type="dot")
    plt.title("Bottom-Up Model SHAP (Home Win Drivers - World Cup)")
    plt.tight_layout()
    plt.savefig(os.path.join(versioned_dir, "shap_bottom_up.png"))
    plt.close()
    
    print(f"✅ SHAP plots successfully saved to {versioned_dir}/")

if __name__ == "__main__":
    main()