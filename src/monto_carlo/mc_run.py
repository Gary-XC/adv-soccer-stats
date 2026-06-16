import pandas as pd
import io
from monto_carlo.mc_simulator import TournamentSimulator

csv_data = """team_name,group
France,A
Germany,A
Morocco,A
Canada,A
Brazil,B
Argentina,B
Japan,B
Cameroon,B
Spain,C
England,C
Ecuador,C
Australia,C
Portugal,D
Netherlands,D
Senegal,D
Qatar,D
Belgium,E
Croatia,E
USA,E
Costa Rica,E
Uruguay,F
Colombia,F
South Korea,F
Ghana,F
Denmark,G
Switzerland,G
Iran,G
Wales,G
Mexico,H
Poland,H
Saudi Arabia,H
Tunisia,H"""

teams_df = pd.read_csv(io.StringIO(csv_data))

# 2. Define Group Stage Fixtures
fixtures_data = []
dummy_features = {col: 0.0 for col in ['td_ewma_shots_on_home', 'td_ewma_defensive_solidity_home', 'td_ewma_goals_total_home', 'td_rest_days_home', 'td_ewma_shots_on_away', 'td_ewma_defensive_solidity_away', 'td_ewma_goals_total_away', 'td_rest_days_away', 'td_elo_diff', 'bu_avg_rating_home', 'bu_sum_shots_home', 'bu_avg_pass_acc_home', 'bu_avg_rating_away', 'bu_sum_shots_away', 'bu_avg_pass_acc_away']}

groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for grp in groups:
    grp_teams = teams_df[teams_df['group'] == grp]['team_name'].tolist()
    matchups = [(grp_teams[0], grp_teams[1]), (grp_teams[2], grp_teams[3]), 
                (grp_teams[0], grp_teams[2]), (grp_teams[1], grp_teams[3]),
                (grp_teams[0], grp_teams[3]), (grp_teams[1], grp_teams[2])]
    for h, a in matchups:
        features = dummy_features.copy()
        features['td_elo_diff'] = 15.0 
        features['td_rest_days_home'] = 5
        features['td_rest_days_away'] = 4
        fixtures_data.append({'home_team': h, 'away_team': a, 'stage': 'Group', 'features': features})
        
fixtures_df = pd.DataFrame(fixtures_data)

simulator = TournamentSimulator("models/stratified_pipeline", n_simulations=10000)
simulator.load_tournament_structure(teams_df)
simulator.run_monte_carlo(fixtures_df)