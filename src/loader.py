# src/loader.py
import os
import glob
import pandas as pd


class DataLoader:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.data = {}

    def load_csvs(self):
        """Scans the designated directory and loads all CSVs into a dictionary."""
        files = glob.glob(os.path.join(self.raw_data_path, "*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_data_path}. Please verify the path is correct."
            )

        for f in files:
            # Cleaning the filename to use as a dictionary key (e.g., 'top_5_leagues_standard.csv' -> 'standard')
            name = os.path.basename(f).replace(".csv", "").replace("top_5_leagues_", "")
            self.data[name] = pd.read_csv(f)
            print(f"Loaded dataset: {name}")

        return self.data
