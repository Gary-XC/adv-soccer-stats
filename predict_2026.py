import pandas as pd
import numpy as np
import joblib
import os
import sys
import requests
from io import StringIO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def fetch_elo_ratings():
    """Fetches live World Football Elo Ratings."""
    try:
        url = "https://www.eloratings.net/World.tsv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df_elo = pd.read_csv(StringIO(response.text), sep='\t', header=None)
        # Column 1 is Country Name, Column 3 is Rating
        elo_map = dict(zip(df_elo.iloc[:, 1].astype(str).str.strip(), pd.to_numeric(df_elo.iloc[:, 3], errors='coerce')))
        return elo_map
    except Exception:
        print("⚠️ Warning: Could not fetch live Elo ratings. Using fallback.")
        return {}

def predict_2026_world_cup():
    print("=== Loading Latest Optimized Models ===")
    with open("models/latest_model_path.txt", "r") as f:
        model_dir = f.read().strip()
    print(f"Using models from: {model_dir}")
    
    # 1. Load models and feature spaces
    model_td = joblib.load(os.path.join(model_dir, "model_top_down.pkl"))
    model_bu = joblib.load(os.path.join(model_dir, "model_bottom_up.pkl"))
    meta_learner = joblib.load(os.path.join(model_dir, "meta_learner.pkl"))
    feature_spaces = joblib.load(os.path.join(model_dir, "feature_spaces.pkl"))
    
    td_cols = feature_spaces['td_cols']
    bu_cols = feature_spaces['bu_cols']
    all_feature_cols = td_cols + bu_cols
    
    # 2. Load historical data to fetch the latest known features for each team
    print("Loading historical feature matrix to extract latest team states...")
    hist_df = pd.read_csv("src/data/hybrid_matrices.csv", parse_dates=['date'])
    hist_df = hist_df.sort_values('date', ascending=False) # Most recent matches first
    
    # 3. Load the 2026 Fixtures    
    fixtures_file = "src/data/builder/2026_world_cup_groups.csv"
    if not os.path.exists(fixtures_file):
        print(f"❌ Error: {fixtures_file} not found. Please ensure your fixtures CSV is in the data/ directory.")
        return
        
    fixtures_df = pd.read_csv(fixtures_file)
    elo_ratings = fetch_elo_ratings()
    global_mean_elo = np.mean(list(elo_ratings.values())) if elo_ratings else 1500.0
    
    predictions = []
    print(f"Generating predictions for {len(fixtures_df)} fixtures...\n")
    
    for idx, row in fixtures_df.iterrows():
        home_team = str(row['Home Team']).strip()
        away_team = str(row['Away Team']).strip()
        match_date = row['Date (Local)']
        
        # --- FEATURE CONSTRUCTION ---
        feature_dict = {col: 0.0 for col in all_feature_cols}
        
        # Helper to get latest features for a team
        def get_latest_features(team_name):
            team_matches = hist_df[(hist_df['team_home'] == team_name) | (hist_df['team_away'] == team_name)]
            if team_matches.empty:
                return None, None
            last_match = team_matches.iloc[0]
            was_home = (last_match['team_home'] == team_name)
            suffix = '_home' if was_home else '_away'
            return last_match, suffix

        home_hist, home_suffix = get_latest_features(home_team)
        away_hist, away_suffix = get_latest_features(away_team)
        
        if home_hist is not None:
            for col in td_cols:
                if col.endswith(home_suffix): feature_dict[col] = home_hist[col]
            for col in bu_cols:
                if col.endswith(home_suffix): feature_dict[col] = home_hist[col]
            feature_dict['td_rest_days_home'] = 7 # Default to 7 days rest for tournament start
            
        if away_hist is not None:
            for col in td_cols:
                if col.endswith(away_suffix): feature_dict[col] = away_hist[col]
            for col in bu_cols:
                if col.endswith(away_suffix): feature_dict[col] = away_hist[col]
            feature_dict['td_rest_days_away'] = 7

        # Calculate Real Elo Diff (Neutral venue = 0 home advantage)
        home_elo = elo_ratings.get(home_team, global_mean_elo)
        away_elo = elo_ratings.get(away_team, global_mean_elo)
        feature_dict['td_elo_diff'] = home_elo - away_elo 
        
        # 4. Generate Probabilities
        X = pd.DataFrame([feature_dict])
        X_td = X[td_cols]
        X_bu = X[bu_cols]
        
        probs_td = model_td.predict_proba(X_td)[0]
        probs_bu = model_bu.predict_proba(X_bu)[0]
        
        meta_features = np.hstack([probs_td, probs_bu]).reshape(1, -1)
        final_probs = meta_learner.predict_proba(meta_features)[0]
        
        outcomes = ['Home Win', 'Draw', 'Away Win']
        predicted_outcome = outcomes[np.argmax(final_probs)]
        confidence = np.max(final_probs) * 100
        
        predictions.append({
            'Round': row['Round'],
            'Date': match_date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Venue': row['Venue'],
            'Prob_Home_Win': f"{final_probs[0]*100:.1f}%",
            'Prob_Draw': f"{final_probs[1]*100:.1f}%",
            'Prob_Away_Win': f"{final_probs[2]*100:.1f}%",
            'Predicted_Outcome': predicted_outcome,
            'Confidence': f"{confidence:.1f}%"
        })
        
    # 5. Save and Display Results
    pred_df = pd.DataFrame(predictions)
    output_path = "src/data/2026_wc_predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to {output_path}\n")
    
    print("--- Top 5 Highest Confidence Predictions ---")
    pred_df['Confidence_Float'] = pred_df['Confidence'].str.rstrip('%').astype(float)
    top_confidence = pred_df.sort_values('Confidence_Float', ascending=False).head(5)
    
    for _, p in top_confidence.iterrows():
        print(f"{p['Round']} | {p['Date']} | {p['Home Team']} vs {p['Away Team']} ➔ {p['Predicted_Outcome']} ({p['Confidence']})")

if __name__ == "__main__":
    predict_2026_world_cup()