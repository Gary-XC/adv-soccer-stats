import matplotlib

matplotlib.use("Agg")  # Required for thread-safety in web servers
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
import seaborn as sns
import os
import pandas as pd

# Define the output directory for AI-generated assets
OUTPUT_DIR = "data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pizza_chart(player_name: str, df: pd.DataFrame):
    """
    Generates a percentile 'Pizza' chart for a specific player compared to
    the rest of the dataset.
    """
    # 1. Define the parameters we want to visualize
    params = [
        "Attacking_USG_pct",
        "TFE",
        "Gls_per_90",
        "Ast_per_90",
        "xG_per_90",
        "xA_per_90",
    ]

    # 2. Filter for the player
    player_data = df[df["Player"] == player_name]
    if player_data.empty:
        return f"Error: Player '{player_name}' not found for visualization."

    # 3. Calculate percentile values for these params
    # We compare the player's value against the entire distribution
    values = []
    for p in params:
        pct = (df[p] < player_data[p].iloc[0]).mean() * 100
        values.append(int(pct))

    # 4. Bake the Pizza Plot
    baker = PyPizza(
        params=params,
        background_color="#EBEBE9",
        straight_line_color="#EBEBE9",
        last_circle_color="#EBEBE9",
        inner_circle_size=5,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(8, 8),
        kwargs_slices=dict(
            facecolor="#1A78CF", edgecolor="#EBEBE9", zorder=2, linewidth=1
        ),
        kwargs_params=dict(color="#000000", fontsize=12, va="center"),
        kwargs_values=dict(
            color="#000000",
            fontsize=12,
            zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="#1A78CF", boxstyle="round,pad=0.2", lw=1
            ),
        ),
    )

    plt.title(f"{player_name} Scouting Profile", size=18, pad=20)

    # 5. Save and Close
    file_name = f"{player_name.replace(' ', '_')}_pizza.png"
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # Critical to prevent memory leaks

    return f"Pizza chart generated: {save_path}"


def plot_usage_scatter(df: pd.DataFrame):
    """
    Creates a macro-view scatter plot: Attacking Usage % vs True Finishing Efficiency.
    Useful for identifying 'Outliers' and 'High-Volume' players.
    """
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Plot the scatter
    scatter = sns.scatterplot(
        data=df, x="Attacking_USG_pct", y="TFE", alpha=0.6, color="#d00000"
    )

    plt.title("League-wide: Attacking Usage vs. Finishing Efficiency", size=15)
    plt.xlabel("Attacking Usage % (Volume)", size=12)
    plt.ylabel("True Finishing Efficiency (Efficiency)", size=12)

    # Save and Close
    save_path = os.path.join(OUTPUT_DIR, "league_usage_scatter.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return f"Scatter plot generated: {save_path}"
