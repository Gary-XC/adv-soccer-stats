import pandas as pd
import numpy as np

import datetime

from src.data_splitter import DataSplitter
from src.stacking_models import StratifiedStackingEnsemble
from src.baseline_engine import EloBaselineEngine
from sklearn.metrics import log_loss, accuracy_score

def main():
    print("=== Phase 2: Configuration-Driven Splitting ===")
    
    match_df = pd.read_csv("src/data/hybrid_matrices.csv", parse_dates=['date'])
    
    splitter = DataSplitter(config_path="config.yaml")
    X_train, y_train, train_df, X_test, y_test, test_df = splitter.split_data(match_df)
    
    print(f"Train set shape: {X_train.shape} | Test set shape: {X_test.shape}")
    y_train_classes = np.ones(len(y_train), dtype=int) # Default to Draw (1)
    y_train_classes[y_train['goals_total_home'] > y_train['goals_total_away']] = 0 # Home Win
    y_train_classes[y_train['goals_total_home'] < y_train['goals_total_away']] = 2 # Away Win
    
    y_test_classes = np.ones(len(y_test), dtype=int)
    y_test_classes[y_test['goals_total_home'] > y_test['goals_total_away']] = 0
    y_test_classes[y_test['goals_total_home'] < y_test['goals_total_away']] = 2

    print("\n=== Phase 3: Stratified Hybrid Stacking ===")
    # Initialize the NEW Stratified Pipeline
    ensemble = StratifiedStackingEnsemble(n_splits=5, random_state=246)
    
    oof_td, oof_bu = ensemble.generate_oof_predictions(X_train, y_train_classes)
    ensemble.train_meta_learner(oof_td, oof_bu, y_train_classes)
    ensemble.fit_final_models(X_train, y_train_classes)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    versioned_dir = f"models/optimized_pipeline_{timestamp}"
    
    print(f"\n=== Saving Versioned Artifacts to: {versioned_dir} ===")
    ensemble.save_artifacts(output_dir=versioned_dir)

    with open("models/latest_model_path.txt", "w") as f:
        f.write(versioned_dir)
    print("✅ 'latest_model_path.txt' updated.")

    print("\n=== Phase 4: Inference & Evaluation ===")
    # Pass X_test to generate calibrated predictions
    test_predictions = ensemble.predict(X_test)
    
    # Evaluate ML Model
    ml_log_loss = log_loss(y_test_classes, test_predictions)
    ml_accuracy = accuracy_score(y_test_classes, np.argmax(test_predictions, axis=1))
    
    print(f"\n--- Final Results ---")
    print(f"Stratified ML Log-Loss: {ml_log_loss:.4f} | Accuracy: {ml_accuracy:.4f}")
    
    # Evaluate Baseline
    baseline = EloBaselineEngine()
    baseline_probs = baseline.predict_proba(test_df)
    baseline_metrics = baseline.evaluate(y_test, baseline_probs)
    print(f"Baseline Elo Log-Loss: {baseline_metrics['log_loss']}")

if __name__ == "__main__":
    main()