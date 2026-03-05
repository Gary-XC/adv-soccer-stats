import os
import glob
import pandas as pd

from src.metrics import (
    calculate_possession_usage,
    calculate_attacking_usage,
    calculate_offensive_goals_added,
    calculate_true_finishing_efficiency,
)


class SoccerPipeline:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.data = {}

    def load_data(self):
        """Scans the directory and loads all CSVs into a dictionary."""
        files = glob.glob(os.path.join(self.raw_data_path, "*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_data_path}. Please check your path."
            )

        for f in files:
            name = os.path.basename(f).replace(".csv", "").replace("top_5_leagues_", "")
            self.data[name] = pd.read_csv(f)
            print(f"Loaded dataset: {name}")

        return self

    def run(self):
        """Executes the custom metrics and merges them into a master DataFrame."""
        print("Calculating advanced metrics from master file...")

        # Grabbing the master dataset
        if "raw_kaggledata" not in self.data:
            raise KeyError(
                "Could not find 'raw_kaggledata.csv' in the data/raw/ folder."
            )

        df_master = self.data["raw_kaggledata"]

        # Feed the master dataframe into all metric functions
        # Since it has all columns, the functions will just extract what they need
        df_possession = calculate_possession_usage(df_master)
        df_usage = calculate_attacking_usage(df_master, df_master)
        df_efficiency = calculate_offensive_goals_added(df_master)
        df_finishing = calculate_true_finishing_efficiency(df_master)

        # Building the Master Comparison Table
        print("Merging into master analytics table...")
        anchor_cols = ["Player", "Squad"]

        # Start with standard info and Attacking Usage
        master_df = df_usage[
            [
                "Player",
                "Nation",
                "Pos",
                "Squad",
                "Comp",
                "Age",
                "90s",
                "Attacking_USG_pct",
            ]
        ].copy()

        # Left merge to preserve all players, adding the new metrics
        master_df = (
            master_df.merge(
                df_possession[anchor_cols + ["Possession_USG_pct"]],
                on=anchor_cols,
                how="left",
            )
            .merge(
                df_efficiency[anchor_cols + ["Total_Offensive_Goals_Added"]],
                on=anchor_cols,
                how="left",
            )
            .merge(df_finishing[anchor_cols + ["TFE"]], on=anchor_cols, how="left")
        )

        # Clean up any potential duplicates
        master_df = master_df.drop_duplicates(subset=anchor_cols)

        return master_df
