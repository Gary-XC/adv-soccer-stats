import numpy as np
import pandas as pd
from collections import defaultdict
import random
import joblib
import os

class Team:
    def __init__(self, name, group):
        self.name = name
        self.group = group
        self.pts = 0
        self.gd = 0
        self.gf = 0
        self.stage_reached = 'Group'

    def reset_stats(self):
        self.pts = 0
        self.gd = 0
        self.gf = 0
        self.stage_reached = 'Group'

class TournamentSimulator:
    def __init__(self, model_artifacts_dir="models/stratified_pipeline", n_simulations=10000):
        self.n_simulations = n_simulations
        
        # 1. Load Stratified Artifacts
        self.model_td = joblib.load(os.path.join(model_artifacts_dir, "model_top_down.pkl"))
        self.model_bu = joblib.load(os.path.join(model_artifacts_dir, "model_bottom_up.pkl"))
        self.meta_learner = joblib.load(os.path.join(model_artifacts_dir, "meta_learner.pkl"))
        
        # 2. Load the exact feature spaces (td_ and bu_ columns)
        feature_spaces = joblib.load(os.path.join(model_artifacts_dir, "feature_spaces.pkl"))
        self.td_cols = feature_spaces['td_cols']
        self.bu_cols = feature_spaces['bu_cols']
        self.all_feature_cols = self.td_cols + self.bu_cols
            
        self.teams = {}
        self.groups = defaultdict(list)
        self.stage_counts = defaultdict(lambda: defaultdict(int))

    def load_tournament_structure(self, teams_df):
        """teams_df should have columns: 'team_name', 'group'"""
        for _, row in teams_df.iterrows():
            team = Team(row['team_name'], row['group'])
            self.teams[team.name] = team
            self.groups[team.group].append(team)

    def predict_match_probs(self, feature_row):
        """
        Routes a single match's features through the Stratified Ensemble.
        feature_row must be a Pandas Series or Dict containing all 15 td_ and bu_ columns.
        """
        X = pd.DataFrame([feature_row])
        
        # Ensure all expected columns exist, fill missing with 0 (safe fallback)
        for col in self.all_feature_cols:
            if col not in X.columns:
                X[col] = 0.0
                
        X_td = X[self.td_cols]
        X_bu = X[self.bu_cols]
        
        # Get Base Model Probabilities
        probs_td = self.model_td.predict_proba(X_td)[0]
        probs_bu = self.model_bu.predict_proba(X_bu)[0]
        
        # Blend via Meta-Learner (Platt Scaling)
        meta_features = np.hstack([probs_td, probs_bu]).reshape(1, -1)
        final_probs = self.meta_learner.predict_proba(meta_features)[0]
        
        return final_probs # [P_Home, P_Draw, P_Away]

    def simulate_scoreline(self, p_h, p_d, p_a):
        """Generates a realistic scoreline based on 1X2 probabilities to track GD/GF."""
        outcome = np.random.choice(['H', 'D', 'A'], p=[p_h, p_d, p_a])
        if outcome == 'H':
            h_goals = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
            a_goals = np.random.randint(0, h_goals)
        elif outcome == 'A':
            a_goals = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
            h_goals = np.random.randint(0, a_goals)
        else: # Draw
            goals = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
            h_goals, a_goals = goals, goals
        return h_goals, a_goals

    def rank_group(self, group_teams):
        """Sorts teams by FIFA Tie-Breakers: Points -> GD -> GF -> Random Draw."""
        return sorted(group_teams, key=lambda t: (t.pts, t.gd, t.gf, random.random()), reverse=True)

    def simulate_knockout_match(self, team1, team2, probs):
        """Eliminates draws and scales probabilities to a binary outcome."""
        p_h, p_d, p_a = probs
        # Prevent division by zero in extreme edge cases
        denom = p_h + p_a
        if denom == 0:
            p1_win, p2_win = 0.5, 0.5
        else:
            p1_win = p_h / denom
            p2_win = p_a / denom
        
        winner = np.random.choice([team1, team2], p=[p1_win, p2_win])
        return winner

    def run_single_iteration(self, fixtures_df):
        """Simulates one full tournament from Group Stage to Final."""
        for team in self.teams.values():
            team.reset_stats()

        # --- GROUP STAGE ---
        group_matches = fixtures_df[fixtures_df['stage'] == 'Group']
        for _, match in group_matches.iterrows():
            h_team = self.teams[match['home_team']]
            a_team = self.teams[match['away_team']]
            
            probs = self.predict_match_probs(match['features'])
            h_goals, a_goals = self.simulate_scoreline(*probs)
            
            h_team.gf += h_goals
            h_team.gd += (h_goals - a_goals)
            a_team.gf += a_goals
            a_team.gd += (a_goals - h_goals)
            
            if h_goals > a_goals: h_team.pts += 3
            elif h_goals < a_goals: a_team.pts += 3
            else:
                h_team.pts += 1
                a_team.pts += 1

        qualified = {}
        for grp, teams in self.groups.items():
            ranked = self.rank_group(teams)
            qualified[f"1{grp}"] = ranked[0]
            qualified[f"2{grp}"] = ranked[1]
            
        for team in self.teams.values():
            self.stage_counts[team.name]['Group'] += 1

        # --- KNOCKOUT STAGE ---
        r16_matchups = [
            ('1A', '2B'), ('1C', '2D'), ('1E', '2F'), ('1G', '2H'),
            ('1B', '2A'), ('1D', '2C'), ('1F', '2E'), ('1H', '2G')
        ]
        
        current_round_teams = []
        for slot1, slot2 in r16_matchups:
            t1, t2 = qualified[slot1], qualified[slot2]
            
            # For knockout stages, we pass a baseline feature vector. 
            # In a live system, you would dynamically calculate the fatigue/EWMA 
            # based on the simulated group stage results.
            dummy_features = {col: 0.0 for col in self.all_feature_cols}
            # Inject a neutral Elo diff for knockouts
            dummy_features['td_elo_diff'] = 0.0 
            # Inject baseline rest days for knockouts (e.g., 4 days between R16 and QF)
            dummy_features['td_rest_days_home'] = 4
            dummy_features['td_rest_days_away'] = 4
            
            probs = self.predict_match_probs(dummy_features)
            winner = self.simulate_knockout_match(t1, t2, probs)
            current_round_teams.append(winner)
            self.stage_counts[winner.name]['Round of 16'] += 1

        stages = ['Quarter-final', 'Semi-final', 'Final']
        for stage_name in stages:
            next_round_teams = []
            for i in range(0, len(current_round_teams), 2):
                t1, t2 = current_round_teams[i], current_round_teams[i+1]
                
                dummy_features = {col: 0.0 for col in self.all_feature_cols}
                dummy_features['td_elo_diff'] = 0.0
                dummy_features['td_rest_days_home'] = 4
                dummy_features['td_rest_days_away'] = 4
                
                probs = self.predict_match_probs(dummy_features)
                winner = self.simulate_knockout_match(t1, t2, probs)
                next_round_teams.append(winner)
                self.stage_counts[winner.name][stage_name] += 1
                
            current_round_teams = next_round_teams

        self.stage_counts[current_round_teams[0].name]['Winner'] += 1

    def run_monte_carlo(self, fixtures_df):
        print(f"Starting {self.n_simulations} Monte Carlo Simulations...")
        for i in range(self.n_simulations):
            if (i + 1) % 2000 == 0:
                print(f"Progress: {i + 1}/{self.n_simulations}")
            self.run_single_iteration(fixtures_df)
            
        self.generate_report()

    def generate_report(self):
        print("\n--- TOURNAMENT PROBABILITIES (%) ---")
        stages = ['Round of 16', 'Quarter-final', 'Semi-final', 'Final', 'Winner']
        
        contenders = [t for t in self.teams.keys() if self.stage_counts[t]['Winner'] > 10]
        contenders.sort(key=lambda t: self.stage_counts[t]['Winner'], reverse=True)
        
        df_report = pd.DataFrame(index=contenders, columns=stages)
        
        for team in contenders:
            for stage in stages:
                prob = (self.stage_counts[team][stage] / self.n_simulations) * 100
                df_report.loc[team, stage] = f"{prob:.2f}%"
                
        print(df_report.to_string())