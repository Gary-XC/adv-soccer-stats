import os
import sys
import pandas as pd
import logging
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_call import fetch_latest_matches
from feature_engineering import generate_hybrid_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DATA_CSV = "data/player_match_metrics_wc_auto.csv"
OUTPUT_MATRIX_CSV = "data/hybrid_matrices.csv"

def process_and_store_data(new_data):
    """Checks existing state and appends uniquely. Returns True if data was added."""
    if not new_data:
        logging.info("No API data returned. Skipping storage.")
        return False

    os.makedirs("data", exist_ok=True)
    existing_ids = set()
    new_df = pd.DataFrame(new_data)
    new_df['fixture_id'] = new_df['fixture_id'].astype(str)

    if os.path.exists(RAW_DATA_CSV):
        existing_df = pd.read_csv(RAW_DATA_CSV, usecols=["fixture_id"])
        existing_ids = set(existing_df['fixture_id'].astype(str).unique())
    
    unique_df = new_df[~new_df['fixture_id'].isin(existing_ids)]

    if unique_df.empty:
        logging.info("Fetched matches already exist in local dataset. Skipping append.")
        return False

    unique_df.to_csv(RAW_DATA_CSV, mode='a', header=not os.path.exists(RAW_DATA_CSV), index=False)
    logging.info(f"Appended {len(unique_df)} new player rows to {RAW_DATA_CSV}")
    return True

def main():
    logging.info("--- Starting Automated Data Update Pipeline ---")
    try:
        # 1. Extract
        new_data = fetch_latest_matches()
        
        # 2. State Check & Load
        data_was_updated = process_and_store_data(new_data)
        
        # 3. Transform (Conditional Gatekeeper)
        if data_was_updated:
            logging.info("--- Regenerating Hybrid Matrices ---")
            generate_hybrid_matrices(data_dir="data", output_path=OUTPUT_MATRIX_CSV)
            logging.info("✅ Pipeline finished successfully!")
        else:
            logging.info("Dataset up-to-date. Skipped matrix regeneration.")
            
    except Exception as e:
        logging.error(f"PIPELINE CRASHED: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()