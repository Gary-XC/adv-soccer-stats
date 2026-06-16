import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.metrics import log_loss

class EloBaselineEngine:
    def __init__(self):
        self.elo_ratings = self._fetch_elo_ratings()
        self.team_to_elo = self._build_team_mapping()

    def _fetch_elo_ratings(self):
        """Fetches the current World Elo Ratings from the TSV endpoint."""
        url = "https://www.eloratings.net/World.tsv"
        try:
            response = requests.get(url)
            response.raise_for_status()
            # The TSV has no header row, so we set header=None
            df = pd.read_csv(StringIO(response.text), sep='\t', header=None)
            
            # Column 2 is Country Code (e.g., 'ES', 'AR'), Column 3 is Current Rating
            elo_df = df[[2, 3]].copy()
            elo_df.columns = ['country_code', 'elo_rating']
            elo_df['elo_rating'] = pd.to_numeric(elo_df['elo_rating'], errors='coerce')
            return elo_df
        except Exception as e:
            print(f"Error fetching Elo ratings: {e}")
            return pd.DataFrame(columns=['country_code', 'elo_rating'])

    def _build_team_mapping(self):
        """
        Maps standard country codes to the team_name strings in your dataset.
        NOTE: You may need to adjust the values (right side) to exactly match 
        the team_name strings in your player_match_metrics CSVs.
        """
        code_to_team = {
            'ES': 'Spain', 'AR': 'Argentina', 'FR': 'France', 'EN': 'England', 
            'BR': 'Brazil', 'PT': 'Portugal', 'NL': 'Netherlands', 'BE': 'Belgium',
            'DE': 'Germany', 'IT': 'Italy', 'UY': 'Uruguay', 'HR': 'Croatia',
            'CO': 'Colombia', 'MX': 'Mexico', 'US': 'United States', 'JP': 'Japan',
            'MA': 'Morocco', 'SN': 'Senegal', 'CH': 'Switzerland', 'DK': 'Denmark',
            'IR': 'Iran', 'KR': 'South Korea', 'AU': 'Australia', 'SA': 'Saudi Arabia',
            'EC': 'Ecuador', 'QA': 'Qatar', 'GH': 'Ghana', 'CM': 'Cameroon',
            'NG': 'Nigeria', 'TN': 'Tunisia', 'CA': 'Canada', 'CR': 'Costa Rica',
            'PL': 'Poland', 'RS': 'Serbia', 'WA': 'Wales', 'AT': 'Austria',
            'CZ': 'Czech Republic', 'TR': 'Turkey', 'RU': 'Russia', 'SE': 'Sweden',
            'UA': 'Ukraine', 'SK': 'Slovakia', 'HU': 'Hungary', 'GR': 'Greece',
            'NO': 'Norway', 'RO': 'Romania', 'SC': 'Scotland', 'FI': 'Finland',
            'IS': 'Iceland', 'CL': 'Chile', 'PE': 'Peru', 'PY': 'Paraguay',
            'VE': 'Venezuela', 'BO': 'Bolivia', 'EG': 'Egypt', 'DZ': 'Algeria',
            'CI': 'Ivory Coast', 'BF': 'Burkina Faso', 'ML': 'Mali', 'ZA': 'South Africa'
        }
        
        team_mapping = {}
        for _, row in self.elo_ratings.iterrows():
            code = row['country_code']
            if code in code_to_team:
                team_mapping[code_to_team[code]] = row['elo_rating']
                
        return team_mapping

    def get_elo(self, team_name):
        """Safely retrieves Elo rating, falling back to global average if team is missing."""
        if not self.team_to_elo:
            return 1500 # Default baseline
        default_elo = np.mean(list(self.team_to_elo.values()))
        return self.team_to_elo.get(team_name, default_elo)

    def predict_proba(self, match_df):
        """
        Generates 1X2 probabilities using the mathematical Elo Expected Score formula.
        """
        predictions = []
        for _, row in match_df.iterrows():
            elo_h = self.get_elo(row['team_home']) 
            elo_a = self.get_elo(row['team_away'])
            
            # Elo Expected Score Formula
            # dr = Rating_Home - Rating_Away + Home_Field_Advantage
            # For World Cup (neutral venues), home field advantage is minimal (~15 points)
            dr = elo_h - elo_a + 15 
            
            prob_h_win = 1 / (1 + 10**(-dr / 400))
            prob_a_win = 1 - prob_h_win
            
            # Heuristic for Draw Probability
            # Draw probability peaks around ~30% when teams are evenly matched (dr ≈ 0)
            # and decays exponentially as the skill gap widens.
            draw_factor = np.exp(-abs(dr) / 150) 
            prob_draw = 0.32 * draw_factor 
            
            # Normalize probabilities so they sum exactly to 1.0
            total = prob_h_win + prob_draw + prob_a_win
            prob_h_win /= total
            prob_draw /= total
            prob_a_win /= total
            
            predictions.append([prob_h_win, prob_draw, prob_a_win])
            
        return np.array(predictions)

    def evaluate(self, y_test_goals, y_pred_proba):
        """Evaluates predictions using Brier Score and Log-Loss."""
        y_true_class = []
        for _, row in y_test_goals.iterrows():
            # Adjust column names based on your actual y_test dataframe
            if row['goals_total_home'] > row['goals_total_away']:
                y_true_class.append(0) # Home Win
            elif row['goals_total_home'] < row['goals_total_away']:
                y_true_class.append(2) # Away Win
            else:
                y_true_class.append(1) # Draw
                
        y_true_class = np.array(y_true_class)
        y_true_onehot = np.eye(3)[y_true_class]
        
        # Brier Score (Multi-class)
        brier_score = np.mean(np.sum((y_true_onehot - y_pred_proba)**2, axis=1))
        
        # Log Loss
        ll = log_loss(y_true_class, y_pred_proba)
        
        return {
            "brier_score": round(brier_score, 4),
            "log_loss": round(ll, 4),
            "matches_evaluated": len(y_true_class)
        }