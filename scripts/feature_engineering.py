import pandas as pd
import numpy as np
import os
import glob
import json
import requests
from io import StringIO

def fetch_elo_ratings():
    """Fetches live World Football Elo Ratings and returns a {Country: Rating} dict."""
    try:
        url = "https://www.eloratings.net/World.tsv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 0=Rank, 1=Country, 2=Code, 3=Rating, 4=Prev, 5=Change
        df_elo = pd.read_csv(StringIO(response.text), sep='\t', header=None)
        
        # Clean country names and map to numeric ratings
        elo_map = dict(zip(
            df_elo.iloc[:, 1].astype(str).str.strip(), 
            pd.to_numeric(df_elo.iloc[:, 3], errors='coerce')
        ))
        print(f"Successfully fetched {len(elo_map)} Elo ratings.")
        return elo_map
    except Exception as e:
        print(f"Warning: Failed to fetch Elo ratings ({e}). Using fallback mean of 1500.")
        return {}

# ==========================================
# MAIN FEATURE ENGINEERING FUNCTION
# ==========================================
def generate_hybrid_matrices(data_dir="data/master", output_path="data/hybrid_matrices.csv"):
    print("=== Starting Hybrid Feature Engineering (Approach 3) ===")
    
    # 1. Ingest Data
    csv_files = glob.glob(os.path.join(data_dir, "player_match_metrics_*.csv"))
    if not csv_files:
        csv_files = glob.glob("player_match_metrics_*.csv")
        
    if not csv_files:
        raise FileNotFoundError("No player_match_metrics_*.csv files found.")
        
    print(f"Ingesting {len(csv_files)} files...")
    df_list = [pd.read_csv(f) for f in csv_files]
    
    for i in range(len(df_list)):
        df_list[i].columns = df_list[i].columns.str.strip()
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Clean dates, numerics, and striping team names
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['team_name'] = df['team_name'].astype(str).str.strip()
    df = df.dropna(subset=['date', 'fixture_id', 'team_id'])
    
    numeric_cols = ['minutes', 'shots_on', 'rating', 'tackles_interceptions', 'passes_total', 'goals_total']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    if 'passes_accuracy' in df.columns:
        df['passes_accuracy'] = df['passes_accuracy'].astype(str).str.replace('%', '', regex=False)
        df['passes_accuracy'] = pd.to_numeric(df['passes_accuracy'], errors='coerce').fillna(0)
    else:
        df['passes_accuracy'] = 0
        
    # Filter only players who played
    df_played = df[df['minutes'] > 0].copy()
    
    # ==========================================
    # BOTTOM-UP: Player-Level EWMA (Talent/Form)
    # ==========================================
    print("Computing Bottom-Up Player EWMA...")
    df_played = df_played.sort_values(['player_id', 'date', 'fixture_id'])
    
    bu_metrics = ['rating', 'shots_on', 'passes_accuracy']
    for col in bu_metrics:
        shifted = df_played.groupby('player_id')[col].shift(1)
        df_played[f'player_ewma_{col}'] = shifted.ewm(span=10, adjust=False).mean()
    
    # 1. Create weighted columns (Metric * Minutes)
    df_played['weighted_rating'] = df_played['player_ewma_rating'] * df_played['minutes']
    df_played['weighted_pass_acc'] = df_played['player_ewma_passes_accuracy'] * df_played['minutes']
        
    # 2. Aggregate to Team-Match Level
    bu_agg = df_played.groupby(['fixture_id', 'team_id']).agg({
        'weighted_rating': 'sum',
        'player_ewma_shots_on': 'sum',
        'weighted_pass_acc': 'sum',
        'minutes': 'sum'  # Total team minutes played in this match
    }).reset_index()
    
    # 3. Calculate the true weighted averages
    bu_agg['bu_avg_rating'] = bu_agg['weighted_rating'] / bu_agg['minutes']
    bu_agg['bu_avg_pass_acc'] = bu_agg['weighted_pass_acc'] / bu_agg['minutes']
    bu_agg['bu_sum_shots'] = bu_agg['player_ewma_shots_on']
    
    # 4. Clean up intermediate columns and keep only the final BU features
    bu_agg = bu_agg[['fixture_id', 'team_id', 'bu_avg_rating', 'bu_sum_shots', 'bu_avg_pass_acc']]
    
    # ==========================================
    # TOP-DOWN: Team-Level EWMA (Cohesion/Tactics)
    # ==========================================
    print("Computing Top-Down Team EWMA...")
    team_stats = df_played.groupby(['fixture_id', 'team_id', 'date', 'league_id', 'league_name', 'team_name']).agg({
        'shots_on': 'sum',
        'tackles_interceptions': 'sum',
        'goals_total': 'sum'
    }).reset_index()
    
    team_stats['defensive_solidity'] = team_stats['tackles_interceptions']
    
    # MUST sort by team and date before calculating diff()
    team_stats = team_stats.sort_values(['team_id', 'date', 'fixture_id'])
    
    # --- ADVANCED FEATURE: REST DAYS (FATIGUE) ---
    team_stats['td_rest_days'] = team_stats.groupby('team_id')['date'].diff().dt.days
    team_stats['td_rest_days'] = team_stats['td_rest_days'].fillna(14)
    team_stats['td_rest_days'] = team_stats['td_rest_days'].clip(upper=30).astype(int)
    
    td_metrics = ['shots_on', 'defensive_solidity', 'goals_total']
    for col in td_metrics:
        shifted = team_stats.groupby('team_id')[col].shift(1)
        team_stats[f'td_ewma_{col}'] = shifted.ewm(span=5, adjust=False).mean()
        
    td_features = team_stats[['fixture_id', 'team_id', 'date', 'league_id', 'league_name', 'team_name', 'goals_total',
                              'td_ewma_shots_on', 'td_ewma_defensive_solidity', 'td_ewma_goals_total', 
                              'td_rest_days']].copy()
    
    # ==========================================
    # MERGE & BUILD MATCH MATRICES
    # ==========================================
    print("Building Stratified Match Matrices...")
    combined = td_features.merge(bu_agg, on=['fixture_id', 'team_id'], how='left')
    
    fixtures = combined.groupby('fixture_id').agg({
        'team_id': list, 
        'date': 'first', 
        'league_id': 'first', 
        'league_name': 'first'
    }).reset_index()
    
    # Keep only valid 2-team fixtures
    fixtures = fixtures[fixtures['team_id'].apply(len) == 2].copy()
    
    # Explicitly create home and away team ID columns
    fixtures['team_id_home'] = fixtures['team_id'].apply(lambda x: x[0])
    fixtures['team_id_away'] = fixtures['team_id'].apply(lambda x: x[1])
    
    # Get unique team names per fixture and team_id for reliable merging
    team_names_df = combined[['fixture_id', 'team_id', 'team_name']].drop_duplicates()
    
    # Merge to get home team name (Vectorized, no apply!)
    fixtures = fixtures.merge(
        team_names_df.rename(columns={'team_id': 'team_id_home', 'team_name': 'team_home'}),
        on=['fixture_id', 'team_id_home'],
        how='left'
    )
    
    # Merge to get away team name (Vectorized, no apply!)
    fixtures = fixtures.merge(
        team_names_df.rename(columns={'team_id': 'team_id_away', 'team_name': 'team_away'}),
        on=['fixture_id', 'team_id_away'],
        how='left'
    )
    
    base_info = fixtures[['fixture_id', 'date', 'league_id', 'league_name', 'team_id_home', 'team_id_away', 'team_home', 'team_away']].copy()
    
    exclude_cols = ['fixture_id', 'team_id', 'date', 'league_id', 'league_name', 'team_name']
    feature_cols = [c for c in combined.columns if c not in exclude_cols]
    
    home_features = combined[['fixture_id', 'team_id'] + feature_cols].copy()
    home_features.rename(columns={c: f"{c}_home" for c in feature_cols}, inplace=True)
    
    home_df = base_info.merge(
        home_features, 
        left_on=['fixture_id', 'team_id_home'], 
        right_on=['fixture_id', 'team_id']
    )
    home_df.drop(columns=['team_id'], inplace=True, errors='ignore')
    
    away_features = combined[['fixture_id', 'team_id'] + feature_cols].copy()
    away_features.rename(columns={c: f"{c}_away" for c in feature_cols}, inplace=True)
    
    away_df = base_info.merge(
        away_features, 
        left_on=['fixture_id', 'team_id_away'], 
        right_on=['fixture_id', 'team_id']
    )
    away_df.drop(columns=['team_id'], inplace=True, errors='ignore')
    
    merge_keys = ['fixture_id', 'date', 'league_id', 'league_name', 'team_id_home', 'team_id_away', 'team_home', 'team_away']
    missing_in_home = [k for k in merge_keys if k not in home_df.columns]
    missing_in_away = [k for k in merge_keys if k not in away_df.columns]
    
    if missing_in_home or missing_in_away:
        raise ValueError(f"Missing merge keys! Home missing: {missing_in_home}, Away missing: {missing_in_away}")
        
    match_df = home_df.merge(away_df, on=merge_keys)
    
    # ==========================================
    # REAL ELO DIFFERENTIAL INTEGRATION
    # ==========================================
    print("Integrating Real Elo Differentials...")
    elo_ratings = fetch_elo_ratings()
    global_mean_elo = np.mean(list(elo_ratings.values())) if elo_ratings else 1500.0
    
    # Map common API-Football names to Elo's official naming convention
    name_mapping = {
        'United States': 'USA', 'South Korea': 'Korea Republic', 'North Korea': 'Korea DPR',
        'DR Congo': 'Congo DR', 'Czech Republic': 'Czechia', 'Republic of Ireland': 'Ireland',
        'China': 'China PR', 'Ivory Coast': "Côte d'Ivoire", 'Cape Verde': 'Cabo Verde',
        'Bosnia': 'Bosnia and Herzegovina', 'Curacao': 'Curaçao', 'Eswatini': 'Swaziland'
    }
    
    def get_elo(team_name):
        clean_name = name_mapping.get(team_name, team_name)
        return elo_ratings.get(clean_name, global_mean_elo) # type: ignore
    
    match_df['home_elo'] = match_df['team_home'].apply(get_elo)
    match_df['away_elo'] = match_df['team_away'].apply(get_elo)
    
    # International/Neutral Tournament IDs (No home advantage)
    neutral_leagues = [1, 4, 9, 5, 28, 11, 10, 12] # World Cup, Euros, Copa America, Nations League, Friendlies, etc.
    
    # Apply +30 Elo points for domestic leagues, +0 for neutral tournaments
    match_df['td_elo_diff'] = np.where(
        match_df['league_id'].isin(neutral_leagues),
        match_df['home_elo'] - match_df['away_elo'],          # Neutral venue (World Cup, etc.)
        match_df['home_elo'] - match_df['away_elo'] + 30      # Domestic league (Home advantage applies)
    )
    
    # Clean up temporary columns
    match_df.drop(columns=['home_elo', 'away_elo'], inplace=True)
    
    # Select final columns to ensure clean CSV
    final_cols = ['fixture_id', 'date', 'league_id', 'league_name', 'team_home', 'team_away', 
                  'goals_total_home', 'goals_total_away']
    
    # Add all td_ and bu_ columns (this automatically catches td_elo_diff, td_rest_days, etc.)
    td_bu_cols = [c for c in match_df.columns if c.startswith('td_') or c.startswith('bu_')]
    final_cols.extend(td_bu_cols)
    
    # Ensure we only select columns that actually exist
    final_cols = [c for c in final_cols if c in match_df.columns]
    match_df = match_df[final_cols]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    match_df.to_csv(output_path, index=False)
    print(f"Successfully saved Hybrid Matrices to {output_path}")
    print(f"Shape: {match_df.shape}")

if __name__ == "__main__":
    generate_hybrid_matrices()