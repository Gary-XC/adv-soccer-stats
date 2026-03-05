import pandas as pd
import numpy as np

def calculate_possession_usage(df):
    """
    Calculates Possession Usage %: The percentage of a team's touches 
    a player accounts for while they are on the pitch.
    Requires: 'top_5_leagues_possession.csv'
    """
    df = df.copy()
    
    # Calculate touches per 90 for the player
    df['Touches_per_90'] = df['Touches'] / df['90s']
    
    # Calculate the team's total touches and total 90s played
    # Note: 11 players are on the pitch, so Team 90s is roughly (Sum of all player 90s / 11)
    team_total_touches = df.groupby('Squad')['Touches'].transform('sum')
    team_total_90s = df.groupby('Squad')['90s'].transform('sum') / 11
    
    # Team touches per 90
    team_touches_per_90 = team_total_touches / team_total_90s
    
    # The Usage Rate (%)
    df['Possession_USG_pct'] = (df['Touches_per_90'] / team_touches_per_90) * 100
    
    return df


def calculate_attacking_usage(df_standard, df_possession):
    """
    Calculates Attacking Usage %: The 'Ball Dominance' metric.
    Dynamically checks for available turnover columns to prevent KeyErrors.
    """
    df_standard = df_standard.copy()
    df_possession = df_possession.copy()
    
    anchor_cols = ['Player', 'Squad']
    # List all possible names for carries and turnovers
    possible_cols = ['Carries', 'CarMis', 'CarDis', 'Mis', 'Dis', 'PrgC'] 
    
    # Keep only the columns that actually exist in the possession dataframe
    available_cols = anchor_cols + [col for col in possible_cols if col in df_possession.columns]
    
    # Merge the standard data with our dynamically filtered possession data
    df = pd.merge(df_standard, df_possession[available_cols], on=['Player', 'Squad'], how='inner')
    
    # .get() will pull the column if it exists, otherwise it returns a Series of 0s
    shots = df.get('Sh', 0)
    
    # Standard usually has PrgC, but we check both dataframes just in case
    prog_carries = df.get('PrgC_x', df.get('PrgC_y', df.get('PrgC', 0))) 
    
    # Handle the different variations of turnover names
    miscontrols = df.get('CarMis', df.get('Mis', 0))
    dispossessed = df.get('CarDis', df.get('Dis', 0))
    
    df['Offensive_Actions'] = shots + prog_carries + miscontrols + dispossessed
    
    # Calculate the Usage Rate
    df['Off_Actions_per_90'] = df['Offensive_Actions'] / df['90s']
    
    team_total_actions = df.groupby('Squad')['Offensive_Actions'].transform('sum')
    team_total_90s = df.groupby('Squad')['90s'].transform('sum') / 11
    
    team_actions_per_90 = team_total_actions / team_total_90s
    
    # Avoid division by zero
    df['Attacking_USG_pct'] = np.where(team_actions_per_90 > 0, 
                                      (df['Off_Actions_per_90'] / team_actions_per_90) * 100, 
                                      0)
    return df


def calculate_offensive_goals_added(df):
    """
    Calculates WAR equivalent: Offensive Goals Added above positional average.
    Requires: 'top_5_leagues_standard.csv'
    """
    df = df.copy()
    
    # Calculate the player's total expected offensive contribution per 90
    df['x_Offense_per_90'] = (df['npxG'] + df.get('xAG', df.get('xA', 0))) / df['90s']
    
    # Calculate the mean expected contribution for their specific position across the whole dataset
    # use transform('mean') to broadcast the positional average back to each player
    df['Positional_Avg_x_Offense'] = df.groupby('Pos')['x_Offense_per_90'].transform('mean')
    
    # Calculate Goals Added (per 90)
    df['Offensive_Goals_Added_per_90'] = df['x_Offense_per_90'] - df['Positional_Avg_x_Offense']
    
    # Calculate Total Goals Added for the season
    df['Total_Offensive_Goals_Added'] = df['Offensive_Goals_Added_per_90'] * df['90s']
    
    return df


def calculate_true_finishing_efficiency(df):
    """
    Calculates True Finishing Efficiency (TFE).
    Requires: 'top_5_leagues_shooting.csv'
    """
    df = df.copy()
    
    # If npG isn't explicitly there, calculate it:
    if 'npG' not in df.columns and 'PK' in df.columns:
        df['npG'] = df['Gls'] - df['PK']
    elif 'npG' not in df.columns:
        df['npG'] = df['Gls'] # Fallback if PK data is missing 
        
    # Calculate TFE, handling division by zero for players with 0 npxG
    # We use np.where to assign np.nan to avoid skewing data with infinite values
    df['TFE'] = np.where(df['npxG'] > 0.1, df['npG'] / df['npxG'], np.nan)
    
    return df