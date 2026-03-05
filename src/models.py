import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class PlayerModels:
    def __init__(self, master_df: pd.DataFrame):
        self.df = master_df.copy()
        self.scaler = StandardScaler()
        self.features = [
            "Possession_USG_pct",
            "Attacking_USG_pct",
            "Total_Offensive_Goals_Added",
            "TFE",
        ]
        self.ml_df = None

    def cluster_roles(self, n_clusters=5):
        """Groups players into usage roles using K-Means clustering."""
        # Filter for players with enough minutes and valid TFE
        ml_df = self.df[(self.df["90s"] >= 5) & (self.df["TFE"].notna())].copy()

        # Replace any hidden infinities with NaN, then dropping the corrupted rows
        ml_df = ml_df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.features)

        # Standardize features
        scaled_features = self.scaler.fit_transform(ml_df[self.features])

        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        ml_df["Player_Role_Cluster"] = kmeans.fit_predict(scaled_features)

        self.ml_df = ml_df
        return ml_df

    def project_season_totals(self):
        """Projects end-of-season goals based on current Usage and TFE."""
        if self.ml_df is None:
            self.cluster_roles()

        df = self.ml_df.copy()  # type: ignore

        # Calculate Goals per 90 proxy and project to a 38-game season
        df["Gls_per_90"] = (df["Attacking_USG_pct"] * df["TFE"]) / 100
        df["Projected_Final_Goals"] = df["Gls_per_90"] * 38

        # Variance = Projected vs Current Added Value
        df["Goal_Variance_Projection"] = (
            df["Projected_Final_Goals"] - df["Total_Offensive_Goals_Added"]
        )

        return df

    def find_statistical_twins(self, player_name: str, num_twins=5):
        """Finds similar players using Euclidean distance on scaled features."""
        if self.ml_df is None:
            self.cluster_roles()

        df = self.ml_df
        target_player = df[df["Player"] == player_name]  # type: ignore

        if target_player.empty:
            return f"Error: Player '{player_name}' not found or lacks sufficient minutes/data."

        # Calculate distance
        source_features = self.scaler.transform(df[self.features])  # type: ignore
        target_features = self.scaler.transform(target_player[self.features])
        distances = cdist(target_features, source_features, metric="euclidean")[0]

        results = df.copy()  # type: ignore
        results["Similarity_Score"] = distances

        # Return the closest matches (making sure to exclude the target player themselves)
        twins = results[results["Player"] != player_name].nsmallest(
            num_twins, "Similarity_Score"
        )
        return twins[["Player", "Squad", "Pos", "Similarity_Score"] + self.features]
