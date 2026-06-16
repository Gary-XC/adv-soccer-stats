import yaml
import pandas as pd

class DataSplitter:
    def __init__(self, config_path="../config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def split_data(self, match_df):
        cutoff_date = pd.to_datetime(self.config['temporal_cutoff'])
        train_leagues = self.config['training_competitions']
        test_leagues = self.config['testing_competitions']
        
        # Ensure date is datetime and tz-naive
        match_df['date'] = pd.to_datetime(match_df['date'], errors='coerce')
        if match_df['date'].dt.tz is not None:
            match_df['date'] = match_df['date'].dt.tz_localize(None)
        match_df = match_df.dropna(subset=['date'])

        # 1. TRAINING DATA: Everything before the cutoff in the training leagues
        train_df = match_df[
            (match_df['date'] < cutoff_date) & 
            (match_df['league_id'].isin(train_leagues))
        ].copy()
        
        # 2. TESTING DATA: ONLY World Cup matches on or after the cutoff
        test_df = match_df[
            (match_df['date'] >= cutoff_date) & 
            (match_df['league_id'].isin(test_leagues))
        ].copy()
        
        # 3. Tag International Matches (for the Stratified Stacking logic)
        international_league_ids = [1, 4, 9, 5, 28, 11, 10, 12]
        train_df['is_international'] = train_df['league_id'].isin(international_league_ids)
        test_df['is_international'] = test_df['league_id'].isin(international_league_ids)
        
        # 4. Feature selection
        feature_cols = [c for c in train_df.columns if c.startswith('td_') or c.startswith('bu_')]
        
        X_train = train_df[feature_cols]
        y_train = train_df[['goals_total_home', 'goals_total_away']] 
        
        X_test = test_df[feature_cols]
        y_test = test_df[['goals_total_home', 'goals_total_away']]
        
        print(f"Data Split Complete:")
        print(f"  - Train Matches (All Leagues): {len(train_df)}")
        print(f"  - Test Matches (World Cup ONLY): {len(test_df)}")
        
        return X_train, y_train, train_df, X_test, y_test, test_df