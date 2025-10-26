import pandas as pd
import requests
import pytz
from typing import Dict, List, Optional, Tuple
from codebase.constants import THEODDS_API_KEY


class ClosingLineOddsFetcher:
    """
    Fetch closing line odds from TheOddsAPI using event IDs.
    Priority: Pinnacle > Betonline.ag > FanDuel
    """
    
    # Default league name to sport_key mapping
    LEAGUE_TO_KEY = {
        "CFL": "americanfootball_cfl",
        "NCAAF": "americanfootball_ncaaf",
        "NCAAF Championship Winner": "americanfootball_ncaaf_championship_winner",
        "NFL": "americanfootball_nfl",
        "NFL Preseason": "americanfootball_nfl_preseason",
        "NFL Super Bowl Winner": "americanfootball_nfl_super_bowl_winner",
        "AFL": "aussierules_afl",
        "KBO": "baseball_kbo",
        "MLB": "baseball_mlb",
        "MLB World Series Winner": "baseball_mlb_world_series_winner",
        "Basketball Euroleague": "basketball_euroleague",
        "NBA": "basketball_nba",
        "NBA Championship Winner": "basketball_nba_championship_winner",
        "NBA Preseason": "basketball_nba_preseason",
        "NBL": "basketball_nbl",
        "NCAAB": "basketball_ncaab",
        "NCAAB Championship Winner": "basketball_ncaab_championship_winner",
        "WNBA": "basketball_wnba",
        "Boxing": "boxing_boxing",
        "ICC Women's World Cup": "cricket_icc_world_cup_womens",
        "One Day Internationals": "cricket_odi",
        "International Twenty20": "cricket_international_t20",
        "Test Matches": "cricket_test_match",
        "Masters Tournament Winner": "golf_masters_tournament_winner",
        "Handball-Bundesliga": "handball_germany_bundesliga",
        "AHL": "icehockey_ahl",
        "Liiga": "icehockey_liiga",
        "Mestis": "icehockey_mestis",
        "NHL": "icehockey_nhl",
        "NHL Championship Winner": "icehockey_nhl_championship_winner",
        "HockeyAllsvenskan": "icehockey_sweden_allsvenskan",
        "SHL": "icehockey_sweden_hockey_league",
        "PLL": "lacrosse_pll",
        "MMA": "mma_mixed_martial_arts",
        "US Presidential Elections Winner": "politics_us_presidential_election_winner",
        "NRL": "rugbyleague_nrl",
        "Primera División - Argentina": "soccer_argentina_primera_division",
        "A-League": "soccer_australia_aleague",
        "Austrian Football Bundesliga": "soccer_austria_bundesliga",
        "Belgium First Div": "soccer_belgium_first_div",
        "Brazil Série A": "soccer_brazil_campeonato",
        "Brazil Série B": "soccer_brazil_serie_b",
        "Primera División - Chile": "soccer_chile_campeonato",
        "Super League - China": "soccer_china_superleague",
        "Copa Libertadores": "soccer_conmebol_copa_libertadores",
        "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
        "Denmark Superliga": "soccer_denmark_superliga",
        "Championship": "soccer_efl_champ",
        "EFL Cup": "soccer_england_efl_cup",
        "League 1": "soccer_england_league1",
        "League 2": "soccer_england_league2",
        "EPL": "soccer_epl",
        "FIFA World Cup Qualifiers - Europe": "soccer_fifa_world_cup_qualifiers_europe",
        "FIFA World Cup Winner": "soccer_fifa_world_cup_winner",
        "Veikkausliiga - Finland": "soccer_finland_veikkausliiga",
        "Ligue 1 - France": "soccer_france_ligue_one",
        "Ligue 2 - France": "soccer_france_ligue_two",
        "Bundesliga - Germany": "soccer_germany_bundesliga",
        "Bundesliga 2 - Germany": "soccer_germany_bundesliga2",
        "3. Liga - Germany": "soccer_germany_liga3",
        "Super League - Greece": "soccer_greece_super_league",
        "Serie A - Italy": "soccer_italy_serie_a",
        "Serie B - Italy": "soccer_italy_serie_b",
        "J League": "soccer_japan_j_league",
        "K League 1": "soccer_korea_kleague1",
        "League of Ireland": "soccer_league_of_ireland",
        "Liga MX": "soccer_mexico_ligamx",
        "Dutch Eredivisie": "soccer_netherlands_eredivisie",
        "Eliteserien - Norway": "soccer_norway_eliteserien",
        "Ekstraklasa - Poland": "soccer_poland_ekstraklasa",
        "Primeira Liga - Portugal": "soccer_portugal_primeira_liga",
        "La Liga - Spain": "soccer_spain_la_liga",
        "La Liga 2 - Spain": "soccer_spain_segunda_division",
        "Premiership - Scotland": "soccer_spl",
        "Allsvenskan - Sweden": "soccer_sweden_allsvenskan",
        "Superettan - Sweden": "soccer_sweden_superettan",
        "Swiss Superleague": "soccer_switzerland_superleague",
        "Turkey Super League": "soccer_turkey_super_league",
        "UEFA Champions League": "soccer_uefa_champs_league",
        "UEFA Champions League Qualification": "soccer_uefa_champs_league_qualification",
        "UEFA Europa League": "soccer_uefa_europa_league",
        "MLS": "soccer_usa_mls",
    }
    
    def __init__(self, api_key: str = THEODDS_API_KEY):
        """
        Initialize the fetcher.
        
        Args:
            api_key: TheOddsAPI key
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.cache = {}  # Cache API responses by (sport_key, event_id, date)
        self.requests_remaining_start = None
        self.requests_remaining_end = None
    
    def convert_league_to_sport_key(self, league: str) -> Optional[str]:
        """
        Convert a league name to a sport_key for the API.
        
        Args:
            league: League name (e.g., 'NBA', 'MLB', 'NCAAF')
            
        Returns:
            sport_key for the API (e.g., 'basketball_nba'), or None if not found
        """
        # Try exact match first
        if league in self.LEAGUE_TO_KEY:
            return self.LEAGUE_TO_KEY[league]
        
        # Try case-insensitive match
        league_upper = league.upper()
        for key, value in self.LEAGUE_TO_KEY.items():
            if key.upper() == league_upper:
                return value
        
        # If already looks like a sport_key, return as-is
        if '_' in league:
            return league
        
        return None
        
    def fetch_event_odds(self, sport_key: str, event_id: str, date: str) -> Optional[Dict]:
        """
        Fetch historical odds for a specific event at a specific time.
        
        Args:
            sport_key: Sport identifier (e.g., 'basketball_nba', 'americanfootball_nfl')
            event_id: The event ID from TheOddsAPI
            date: ISO timestamp for the odds snapshot (e.g., '2023-11-29T22:45:00Z')
            
        Returns:
            Dictionary with odds data, or None if error
        """
        cache_key = (sport_key, event_id, date)
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"{self.base_url}/historical/sports/{sport_key}/events/{event_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'date': date
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self.cache[cache_key] = data
            
            # Track API usage
            requests_remaining = response.headers.get("x-requests-remaining")
            if requests_remaining:
                if self.requests_remaining_start is None:
                    self.requests_remaining_start = int(requests_remaining)
                self.requests_remaining_end = int(requests_remaining)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching odds for event {event_id}: {e}")
            return None
    
    def get_closing_line_odds(self, sport_key: str, event_id: str, 
                              team_name: str, game_time: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Get closing line odds for a specific team with bookmaker priority.
        
        Args:
            sport_key: Sport identifier
            event_id: The event ID from TheOddsAPI
            team_name: The team name to get odds for
            game_time: Game start time ISO timestamp to get odds near game start
            
        Returns:
            Tuple of (closing_odds, bookmaker_name, timestamp) or (None, None, None) if not found
        """
        # Use the game time as the date parameter to get odds closest to game start
        odds_data = self.fetch_event_odds(sport_key, event_id, game_time)
        
        if not odds_data or 'data' not in odds_data:
            return None, None, None
        
        # Extract timestamp from the API response
        snapshot_timestamp = odds_data.get('timestamp')
        
        # The API returns a single snapshot for the specified date
        event_data = odds_data['data']
        bookmakers = event_data.get('bookmakers', [])
        
        if not bookmakers:
            return None, None, snapshot_timestamp
        
        # Priority order for bookmakers
        priority_bookmakers = [
            ('pinnacle', 'Pinnacle'),
            ('fanduel', 'FanDuel'),
            ('draftkings', 'DraftKings')
        ]
        
        # Try priority bookmakers first
        for bookmaker_key, bookmaker_name in priority_bookmakers:
            odds = self._extract_odds_from_bookmaker(bookmakers, bookmaker_key, team_name)
            if odds is not None:
                return odds, bookmaker_name, snapshot_timestamp
        
        # If none of the priority bookmakers have odds, use any available bookmaker
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get('key')
            bookmaker_title = bookmaker.get('title', bookmaker_key)
            odds = self._extract_odds_from_bookmaker(bookmakers, bookmaker_key, team_name)
            if odds is not None:
                return odds, bookmaker_title, snapshot_timestamp
        
        return None, None, snapshot_timestamp
    
    def _extract_odds_from_bookmaker(self, bookmakers: List[Dict], 
                                     bookmaker_key: str, team_name: str) -> Optional[float]:
        """
        Extract odds for a team from a specific bookmaker.
        
        Args:
            bookmakers: List of bookmaker data
            bookmaker_key: The bookmaker key to search for
            team_name: The team name to get odds for
            
        Returns:
            Odds value or None if not found
        """
        for bookmaker in bookmakers:
            if bookmaker.get('key') == bookmaker_key:
                markets = bookmaker.get('markets', [])
                for market in markets:
                    if market.get('key') == 'h2h':
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            # Match team name (case-insensitive)
                            if outcome.get('name', '').lower() == team_name.lower():
                                return outcome.get('price')
        return None
    
    def add_closing_odds_to_df(self, df: pd.DataFrame,
                               event_id_col: str = 'Event ID',
                               team_col: str = 'Team',
                               league_col: str = 'League',
                               event_commence_time_col: str = 'Event ID Commence Time',
                               scrape_time_col: str = 'Scrape Time') -> pd.DataFrame:
        """
        Add closing line odds to the dataframe.
        
        Args:
            df: Input dataframe with event IDs
            event_id_col: Column name containing event IDs (default 'Event ID')
            team_col: Column name containing team names (default 'Team')
            league_col: Column name containing league/sport key (default 'League')
            event_commence_time_col: Column name containing event commence time (default 'Event ID Commence Time')
            scrape_time_col: Column name containing scrape time (default 'Scrape Time')
            
        Returns:
            DataFrame with added 'Closing Line Odds', 'Closing Line Book', and 'API Snapshot Timestamp' columns
        """
        df = df.copy()
        
        # Handle empty dataframe
        if len(df) == 0:
            if 'Closing Line Odds' not in df.columns:
                df['Closing Line Odds'] = None
            if 'Closing Line Book' not in df.columns:
                df['Closing Line Book'] = None
            if 'API Snapshot Timestamp' not in df.columns:
                df['API Snapshot Timestamp'] = None
            return df
        
        # Only create columns if they don't exist
        if 'Closing Line Odds' not in df.columns:
            df['Closing Line Odds'] = None
        if 'Closing Line Book' not in df.columns:
            df['Closing Line Book'] = None
        if 'API Snapshot Timestamp' not in df.columns:
            df['API Snapshot Timestamp'] = None
        
        # Filter to rows that need closing odds (where closing odds is empty/null)
        rows_needing_odds = df['Closing Line Odds'].isna()
        df_to_process = df[rows_needing_odds].copy()
        
        # Also filter out rows with no event ID
        df_to_process = df_to_process[df_to_process[event_id_col].notna()]
        
        if len(df_to_process) == 0:
            print("No rows need closing line odds or all rows missing Event IDs.")
            return df
        print(f"Fetching closing line odds for {len(df_to_process)} rows...")
        print(f"Skipping {(~rows_needing_odds).sum()} rows that already have closing odds")
        print("-" * 70)
        successful_fetches = 0
        skipped_scrape_time = 0
        for idx, row in df_to_process.iterrows():
            event_id = row[event_id_col]
            team_name = row[team_col]
            league_name = row[league_col]
            event_commence_time = row[event_commence_time_col]
            scrape_time = row.get(scrape_time_col)
            if pd.isna(event_id) or pd.isna(team_name) or pd.isna(league_name):
                continue
            
            # Skip if no commence time available
            if pd.isna(event_commence_time):
                print(f"  ⚠ Skipping {team_name}: No event commence time available")
                continue
            
            # Convert league name to sport_key
            sport_key = self.convert_league_to_sport_key(league_name)
            if sport_key is None:
                print(f"  ⚠ Could not map league '{league_name}' to sport key. Skipping.")
                continue
            
            # Convert event commence time to datetime
            try:
                commence_time_dt = pd.to_datetime(event_commence_time)
            except:
                print(f"    ⚠ Could not parse event commence time: {event_commence_time}")
                continue
            
            # Check if scrape time is after game start time
            if not pd.isna(scrape_time):
                try:
                    # Parse scrape time and assume it's in EST/EDT
                    scrape_time_dt = pd.to_datetime(scrape_time)
                    
                    # If scrape time is naive (no timezone), localize it to EST
                    if scrape_time_dt.tzinfo is None:
                        est = pytz.timezone('America/New_York')
                        scrape_time_dt = est.localize(scrape_time_dt)
                    
                    # Convert scrape time to UTC for comparison
                    scrape_time_utc = scrape_time_dt.astimezone(pytz.UTC)
                    
                    # Ensure commence_time_dt is timezone-aware (should already be UTC from API)
                    if commence_time_dt.tzinfo is None:
                        commence_time_dt = pytz.UTC.localize(commence_time_dt)
                    
                    # If scrape happened AFTER game start, skip this row (no closing line available)
                    if scrape_time_utc > commence_time_dt:
                        print(f"  ⚠ Skipping {team_name}: Scrape time ({scrape_time_utc}) after game start ({commence_time_dt})")
                        skipped_scrape_time += 1
                        continue
                except Exception as e:
                    print(f"    ⚠ Could not parse scrape time: {scrape_time} - {e}")
            
            # Convert commence time to ISO format with Z suffix
            commence_time_iso = commence_time_dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
            
            print(f"  Fetching odds for: {team_name} (Event: {event_id[:8]}...)")
            
            closing_odds, bookmaker_used, snapshot_timestamp = self.get_closing_line_odds(
                sport_key, event_id, team_name, commence_time_iso
            )
            
            if closing_odds is not None:
                df.loc[idx, 'Closing Line Odds'] = closing_odds
                df.loc[idx, 'Closing Line Book'] = bookmaker_used
                df.loc[idx, 'API Snapshot Timestamp'] = snapshot_timestamp
                successful_fetches += 1
                print(f"    ✓ Found: {closing_odds} ({bookmaker_used})")
            else:
                # Even if no odds found, store timestamp if available
                if snapshot_timestamp is not None:
                    df.loc[idx, 'API Snapshot Timestamp'] = snapshot_timestamp
                print(f"    ⚠ No closing odds found")
        
        print("-" * 70)
        success_pct = (successful_fetches / len(df_to_process)) * 100 if len(df_to_process) > 0 else 0
        print(f"✓ Successfully fetched: {successful_fetches}/{len(df_to_process)} ({success_pct:.1f}%)")
        if skipped_scrape_time > 0:
            print(f"⚠ Skipped {skipped_scrape_time} rows where scrape time was before game start")
        
        # Print API usage summary
        if self.requests_remaining_start is not None and self.requests_remaining_end is not None:
            requests_used = self.requests_remaining_start - self.requests_remaining_end
            print("\n" + "=" * 70)
            print("API USAGE SUMMARY:")
            print(f"  Requests at start: {self.requests_remaining_start}")
            print(f"  Requests at end: {self.requests_remaining_end}")
            print(f"  Total requests used: {requests_used}")
            print("=" * 70)
        
        return df


# Example usage
if __name__ == "__main__":
    # Load your dataframe
    df = pd.read_csv("codebase/data/clv_mini.csv")
    
    print("Input DataFrame:")
    print(df.head())
    print(f"Total rows: {len(df)}")
    print("\n" + "=" * 70 + "\n")
    
    # Initialize fetcher
    fetcher = ClosingLineOddsFetcher(api_key=THEODDS_API_KEY)
    
    # Add closing line odds to dataframe
    df_with_closing = fetcher.add_closing_odds_to_df(
        df,
        event_id_col='Event ID',
        team_col='Team',
        league_col='League',
        event_commence_time_col='Event ID Commence Time',
        scrape_time_col='Scrape Time'
    )
    
    print("\n" + "=" * 70)
    print("\nResulting DataFrame:")
    print(df_with_closing[['Match', 'Team', 'Closing Line Odds', 'Closing Line Book']].head(10))
    print(f"\nRows with closing odds: {df_with_closing['Closing Line Odds'].notna().sum()}/{len(df_with_closing)}")
    
    # Save to CSV
    output_path = "codebase/data/with_closing_odds.csv"
    df_with_closing.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")