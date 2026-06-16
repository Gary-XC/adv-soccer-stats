import os
import time
from datetime import datetime, timedelta
import requests

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io" 
HEADERS = {"x-apisports-key": API_KEY}
WORLD_CUP_LEAGUE_ID = 1
SEASON = 2026

def _make_request(url, params):
    """Internal helper for safe, rate-limited requests."""
    time.sleep(0.15) 
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("errors") and "requests" in str(data["errors"]).lower():
                    print("\n[Warning] API Rate Limit hit. Sleeping 60s...")
                    time.sleep(60)
                    continue
                return data
                
            if response.status_code == 429:
                print(f"\n[Warning] 429 hit. Sleeping 60s... (Attempt {attempt+1}/{max_attempts})")
                time.sleep(60)
                continue
                
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            print(f"\n[Error] Network error: {e}. Retrying in 5s...")
            time.sleep(5)
                
    return {}

def fetch_latest_matches(target_date=None):
    """Fetches World Cup matches for a given date, extracting active players."""
    if not target_date:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    print(f"Fetching World Cup schedule for {target_date}...")
    
    # 1. Get the fixture IDs for that specific date
    schedule_data = _make_request(f"{BASE_URL}/fixtures", params={
        "league": WORLD_CUP_LEAGUE_ID, 
        "season": SEASON,
        "date": target_date
    })
    
    fixture_ids = []
    if schedule_data and schedule_data.get("response"):
        fixture_ids = [str(match["fixture"]["id"]) for match in schedule_data["response"]]
        
    if not fixture_ids:
        print("No World Cup matches found for this date.")
        return []

    # 2. Fetch detailed stats in one batch request
    print(f"Found {len(fixture_ids)} matches. Fetching detailed player stats...")
    details_data = _make_request(f"{BASE_URL}/fixtures", params={"ids": "-".join(fixture_ids)})
    
    if not details_data or not details_data.get("response"):
        return []

    # 3. Extract metrics for players who played > 0 minutes
    flat_rows = []
    for detailed_fixture in details_data["response"]:
        fixture_info = detailed_fixture.get("fixture", {})
        league_info = detailed_fixture.get("league", {})
        
        for team_data in detailed_fixture.get("players", []):
            for player_stat in team_data.get("players", []):
                stats = player_stat.get("statistics", [{}])[0]
                minutes = stats.get("games", {}).get("minutes")
                
                if minutes is not None and int(minutes) > 0:
                    row = {
                        "player_id": str(player_stat["player"]["id"]),
                        "player_name": player_stat["player"]["name"],
                        "fixture_id": str(fixture_info.get("id")),
                        "date": fixture_info.get("date"),
                        "league_id": league_info.get("id"),
                        "league_name": league_info.get("name"),
                        "team_id": team_data["team"]["id"],
                        "team_name": team_data["team"]["name"],
                        "minutes": minutes,
                        "number": stats.get("games", {}).get("number"),
                        "position": stats.get("games", {}).get("position"),
                        "rating": stats.get("games", {}).get("rating"),
                        "captain": stats.get("games", {}).get("captain"),
                        "substitute": stats.get("games", {}).get("substitute"),
                        "offsides": stats.get("offsides"),
                        "shots_total": stats.get("shots", {}).get("total"),
                        "shots_on": stats.get("shots", {}).get("on"),
                        "goals_total": stats.get("goals", {}).get("total"),
                        "goals_conceded": stats.get("goals", {}).get("conceded"),
                        "goals_assists": stats.get("goals", {}).get("assists"),
                        "goals_saves": stats.get("goals", {}).get("saves"),
                        "passes_total": stats.get("passes", {}).get("total"),
                        "passes_key": stats.get("passes", {}).get("key"),
                        "passes_accuracy": stats.get("passes", {}).get("accuracy"),
                        "tackles_total": stats.get("tackles", {}).get("total"),
                        "tackles_blocks": stats.get("tackles", {}).get("blocks"),
                        "tackles_interceptions": stats.get("tackles", {}).get("interceptions"),
                        "duels_total": stats.get("duels", {}).get("total"),
                        "duels_won": stats.get("duels", {}).get("won"),
                        "dribbles_attempts": stats.get("dribbles", {}).get("attempts"),
                        "dribbles_success": stats.get("dribbles", {}).get("success"),
                        "dribbles_past": stats.get("dribbles", {}).get("past"),
                        "fouls_drawn": stats.get("fouls", {}).get("drawn"),
                        "fouls_committed": stats.get("fouls", {}).get("committed"),
                        "cards_yellow": stats.get("cards", {}).get("yellow"),
                        "cards_red": stats.get("cards", {}).get("red"),
                        "penalty_won": stats.get("penalty", {}).get("won"),
                        "penalty_commited": stats.get("penalty", {}).get("commited"),
                        "penalty_scored": stats.get("penalty", {}).get("scored"),
                        "penalty_missed": stats.get("penalty", {}).get("missed"),
                        "penalty_saved": stats.get("penalty", {}).get("saved")
                    }
                    flat_rows.append(row)
                    
    return flat_rows