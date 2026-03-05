import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import SoccerPipeline
from src.models import PlayerModels
from src.visualization import plot_pizza_chart, plot_usage_vs_efficiency


def main():
    print("Starting Soccer Analytics Pipeline...")

    # Directories
    RAW_PATH = "data/historical"
    PROCESSED_PATH = "data/processed"
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # Running the Data Extraction & Engineering Pipeline
    print("\n--- Phase 1: Data Processing ---")
    pipeline = SoccerPipeline(RAW_PATH)
    pipeline.load_data()
    master_df = pipeline.run()

    # Machine Learning Models
    print("\n--- Phase 2: Machine Learning ---")
    player_models = PlayerModels(master_df)

    print("Clustering player roles...")
    player_models.cluster_roles(n_clusters=5)

    print("Calculating season projections...")
    final_analytics_df = player_models.project_season_totals()

    # Exporting the finalized data
    print("\n--- Phase 3: Export ---")
    output_file = os.path.join(PROCESSED_PATH, "master_analytics.csv")
    final_analytics_df.to_csv(output_file, index=False)
    print(f"Success! Master analytics dataset saved to {output_file}")

    # Example to make sure that everything is working
    GEN_IMAGES_PATH = "visuals"
    print("\n--- Phase 4: Verification Demo ---")
    target_player = "Vinicius Júnior"

    try:
        print(f"\nFinding statistical twins for {target_player}:")
        twins = player_models.find_statistical_twins(target_player, num_twins=3)
        print(
            twins[["Player", "Squad", "Pos", "Similarity_Score"]]  # type: ignore
        )  # pyright: ignore[reportArgumentType]
    except Exception as e:
        print(f"Could not find twins for {target_player}. Error: {e}")

    # Saving the visualizations
    print(f"\nGenerating and saving pizza chart for {target_player}...")
    pizza_fig = plot_pizza_chart(target_player, final_analytics_df)
    if pizza_fig:
        safe_name = target_player.replace(" ", "_").lower()
        pizza_path = os.path.join(GEN_IMAGES_PATH, f"{safe_name}_pizza.png")
        pizza_fig.savefig(pizza_path, bbox_inches="tight", dpi=300)

    print("\nGenerating and saving Volume vs Efficiency scatter plot...")
    scatter_fig = plot_usage_vs_efficiency(final_analytics_df, top_n=50)
    if scatter_fig:
        scatter_path = os.path.join(GEN_IMAGES_PATH, "volume_vs_efficiency_top50.png")
        scatter_fig.savefig(scatter_path, bbox_inches="tight", dpi=300)

    print(f"\nAll visualizations successfully saved to: {GEN_IMAGES_PATH}")


if __name__ == "__main__":
    main()
