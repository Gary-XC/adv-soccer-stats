import os
import requests
import csv
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY
}

def export_group_stages_to_csv(filename="2026_world_cup_groups.csv"):
    url = f"{BASE_URL}/fixtures"
    querystring = {"league": "1", "season": "2026"}
    
    response = requests.get(url, headers=HEADERS, params=querystring)
    
    if response.status_code != 200:
        print(f"Failed to fetch API data: HTTP {response.status_code}")
        return
        
    fixtures = response.json().get("response", [])
    
    # Isolate only the group stage matches
    group_fixtures = [f for f in fixtures if "Group" in f['league']['round']]
    
    if not group_fixtures:
        print("No group stage fixtures found. The API may not be fully populated yet.")
        return

    # Write to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["Round", "Date (Local)", "Time (UTC)", "Home Team", "Away Team", "Venue", "Status"])
        
        for f in group_fixtures:
            round_name = f['league']['round']
            home_team = f['teams']['home']['name']
            away_team = f['teams']['away']['name']
            
            # API can sometimes return null for venue before it is finalized
            venue_data = f['fixture']['venue']
            venue_name = venue_data['name'] if venue_data and venue_data['name'] else "TBD"
            
            status = f['fixture']['status']['short']
            raw_date = f['fixture']['date']
            
            try:
                # Convert "2026-06-11T16:00:00+00:00" to readable columns
                dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d')
                time_str = dt.strftime('%H:%M:%S')
            except ValueError:
                date_str = raw_date
                time_str = "Unknown"
                
            writer.writerow([round_name, date_str, time_str, home_team, away_team, venue_name, status])
            
    print(f"Success: {len(group_fixtures)} group stage matches saved to {filename}")

if __name__ == "__main__":
    export_group_stages_to_csv()