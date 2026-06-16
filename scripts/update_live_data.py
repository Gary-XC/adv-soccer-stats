import os
import pandas as pd
import requests
import glob

# 1. FETCH NEW DATA 
def fetch_latest_matches():
    return [] 

def append_raw_data(new_matches):
    if not new_matches:
        print("No new matches to process.")
        return
    print("Appended new raw data.")

# 3. REGENERATE HYBRID MATRICES
def regenerate_matrices():
    print("Regenerating hybrid_matrices.csv...")
    from feature_engineering import generate_hybrid_matrices
    generate_hybrid_matrices(data_dir="data", output_path="data/hybrid_matrices.csv")
    print("✅ Successfully regenerated hybrid_matrices.csv!")

if __name__ == "__main__":
    new_data = fetch_latest_matches()
    append_raw_data(new_data)
    regenerate_matrices()