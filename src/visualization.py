# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import PyPizza
import pandas as pd

def plot_pizza_chart(player_name: str, df: pd.DataFrame):
    """Generates a percentile pizza chart for a player vs positional peers."""
    player_row = df[df['Player'] == player_name]
    if player_row.empty:
        print(f"Player {player_name} not found in the dataset.")
        return
    
    player_pos = player_row.iloc[0]['Pos']
    peers = df[df['Pos'] == player_pos].copy()
    
    metrics = ['Possession_USG_pct', 'Attacking_USG_pct', 'Total_Offensive_Goals_Added', 'TFE']
    percentiles = []
    
    for m in metrics:
        # Calculate rank percentage within positional peers
        rank = peers[m].rank(pct=True).loc[df['Player'] == player_name].values[0]
        percentiles.append(int(rank * 100))
        
    baker = PyPizza(
        params=["Possession USG%", "Attacking USG%", "Offensive Goals Added", "True Finishing"],
        background_color="#222222", straight_line_color="#000000",
        straight_line_lw=1, last_circle_lw=0, other_circle_lw=0, inner_circle_size=20
    )

    fig, ax = baker.make_pizza(
        percentiles, figsize=(8, 8), param_location=110,
        kwargs_slices=dict(facecolor="#1A78CF", edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color="#F2F2F2", fontsize=12, va="center"),
        kwargs_values=dict(
            color="#000000", fontsize=11, zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="#1A78CF", boxstyle="round,pad=0.2", lw=1)
        )
    )
    fig.text(0.515, 0.97, f"{player_name} - Positional Percentiles", size=16, ha="center", color="#F2F2F2")
    
    plt.show()
    return fig

def plot_usage_vs_efficiency(df: pd.DataFrame, top_n=50):
    """Plots Volume vs Efficiency scatter plot for the top N usage players."""
    # Filter and grab top N active players
    top_df = df[df['90s'] >= 5].nlargest(top_n, 'Attacking_USG_pct')
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=top_df, x='Attacking_USG_pct', y='TFE', 
        size='90s', sizes=(100, 500), alpha=0.6, edgecolor='white'
    )
    
    # Add text labels offset slightly from the dots
    for i, row in top_df.iterrows():
        plt.text(row['Attacking_USG_pct'] + 0.2, row['TFE'], row['Player'], fontsize=9, alpha=0.9)
        
    plt.title(f"Volume vs Efficiency: Top {top_n} Attacking Players", fontsize=15, pad=20)
    plt.xlabel("Attacking Usage % (Volume)", fontsize=12)
    plt.ylabel("True Finishing Efficiency (Quality)", fontsize=12)
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Avg Efficiency')
    plt.legend(title='90s Played', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()