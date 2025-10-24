import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional
import pytz
from codebase.constants import THEODDS_API_KEY

class OddsAPIEventMatcher:
    """
    Efficiently fetch and match event IDs from TheOddsAPI to a dataframe of sports bets.
    Matches based on "Away Team @ Home Team" format and exact start times.
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
        "NBA": "basketball_nba",
        "NBA Championship Winner": "basketball_nba_championship_winner",
        "NCAAB": "basketball_ncaab",
        "NCAAB Championship Winner": "basketball_ncaab_championship_winner",
        "WNBA": "basketball_wnba",
        "Boxing": "boxing_boxing",
        "International Twenty20": "cricket_international_t20",
        "Test Matches": "cricket_test_match",
        "Masters Tournament Winner": "golf_masters_tournament_winner",
        "NHL": "icehockey_nhl",
        "NHL Championship Winner": "icehockey_nhl_championship_winner",
        "PLL": "lacrosse_pll",
        "MMA": "mma_mixed_martial_arts",
        "NRL": "rugbyleague_nrl",
        "Primera División - Argentina": "soccer_argentina_primera_division",
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
        "Super League - Greece": "soccer_greece_super_league",
        "Serie A - Italy": "soccer_italy_serie_a",
        "J League": "soccer_japan_j_league",
        "K League 1": "soccer_korea_kleague1",
        "League of Ireland": "soccer_league_of_ireland",
        "Liga MX": "soccer_mexico_ligamx",
        "Dutch Eredivisie": "soccer_netherlands_eredivisie",
        "Eliteserien - Norway": "soccer_norway_eliteserien",
        "Ekstraklasa - Poland": "soccer_poland_ekstraklasa",
        "La Liga - Spain": "soccer_spain_la_liga",
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
    
    def __init__(self, api_key: str = THEODDS_API_KEY, timezone: str = 'America/New_York', 
                 league_mapping: Dict[str, str] = None):
        """
        Initialize the matcher.
        
        Args:
            api_key: TheOddsAPI key
            timezone: Timezone of the start times in your dataframe (default 'America/New_York' for EST/EDT)
            league_mapping: Custom mapping of league names to sport_keys. If None, uses LEAGUE_TO_KEY
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.timezone = pytz.timezone(timezone)
        self.cache = {}  # Cache API responses by (sport_key, date)
        self.league_mapping = league_mapping if league_mapping is not None else self.LEAGUE_TO_KEY.copy()
    
    def convert_league_to_sport_key(self, league: str) -> Optional[str]:
        """
        Convert a league name to a sport_key for the API.
        
        Args:
            league: League name (e.g., 'NBA', 'MLB')
            
        Returns:
            sport_key for the API (e.g., 'basketball_nba'), or None if not found
        """
        # Try exact match first
        if league in self.league_mapping:
            return self.league_mapping[league]
        
        # Try case-insensitive match
        league_upper = league.upper()
        for key, value in self.league_mapping.items():
            if key.upper() == league_upper:
                return value
        
        # If already looks like a sport_key, return as-is
        if '_' in league:
            return league
        
        return None
        
    def fetch_historical_events(self, sport_key: str, date: str) -> List[Dict]:
        """
        Fetch historical events for a specific sport and date.
        
        Args:
            sport_key: Sport identifier (e.g., 'basketball_nba', 'baseball_mlb')
            date: Date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
            
        Returns:
            List of event dictionaries from the API
        """
        cache_key = (sport_key, date)
        
        # Return cached result if available
        if cache_key in self.cache:
            print(f"  Using cached data for {sport_key} on {date}")
            return self.cache[cache_key]
        
        url = f"{self.base_url}/historical/sports/{sport_key}/events"
        params = {
            'apiKey': self.api_key,
            'date': date,
        }
        
        try:
            print(f"  Fetching events for {sport_key} on {date}...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            events = response.json().get('data', [])
            self.cache[cache_key] = events
            
            print(f"    → Found {len(events)} events")
            
            return events
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching data for {sport_key} on {date}: {e}")
            return []
    
    def parse_match_string(self, match_str: str) -> tuple:
        """
        Parse match string in "Away Team @ Home Team" format.
        
        Args:
            match_str: Match string like "Milwaukee Brewers @ San Diego Padres"
            
        Returns:
            Tuple of (away_team, home_team)
        """
        if '@' not in match_str:
            return None, None
        
        parts = match_str.split('@')
        away_team = parts[0].strip()
        home_team = parts[1].strip()
        
        return away_team, home_team
    
    def convert_to_utc(self, dt: datetime) -> datetime:
        """
        Convert a datetime from the configured timezone to UTC.
        
        Args:
            dt: Datetime object (naive, assumed to be in self.timezone)
            
        Returns:
            Datetime in UTC
        """
        # If datetime is naive, localize it to the configured timezone
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        
        # Convert to UTC
        return dt.astimezone(pytz.UTC)
    
    def match_event(self, away_team: str, home_team: str, start_time_utc: datetime, 
                    events: List[Dict]) -> Optional[str]:
        """
        Match teams and time to an event from the API response.
        
        Args:
            away_team: Away team name
            home_team: Home team name
            start_time_utc: Start time in UTC
            events: List of events from the API
            
        Returns:
            Event ID if match found, None otherwise
        """
        for event in events:
            event_home = event.get('home_team', '')
            event_away = event.get('away_team', '')
            commence_time = event.get('commence_time')
            
            # Check if teams match (case-insensitive)
            teams_match = (
                event_home.lower() == home_team.lower() and 
                event_away.lower() == away_team.lower()
            )
            
            if not teams_match:
                continue
            
            # Check if times match
            try:
                event_time = pd.to_datetime(commence_time).tz_convert(pytz.UTC)
                
                # Times must match exactly (within 1 minute for rounding differences)
                time_diff = abs((start_time_utc - event_time).total_seconds())
                
                if time_diff <= 600*6*4:  # Within 1 minute
                    return event.get('id')
                    
            except Exception as e:
                print(f"    ⚠ Error parsing time for event: {e}")
                continue
        
        return None
    
    def add_event_ids_to_df(self, df: pd.DataFrame, 
                           sport_key_col: str = 'League',
                           start_time_col: str = 'Start Time',
                           match_col: str = 'Match') -> pd.DataFrame:
        """
        Add event IDs to the dataframe.
        
        Args:
            df: Input dataframe with betting data
            sport_key_col: Column name containing sport/league (default 'League')
            start_time_col: Column name containing start time in EST (default 'Start Time')
            match_col: Column name containing match in "Away @ Home" format (default 'Match')
            
        Returns:
            DataFrame with added 'event_id' column
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Parse match column to get away and home teams
        print("Parsing match strings...")

        # Handle empty dataframe
        if len(df) == 0:
            df['event_id'] = None
            return df
        
        df[['away_team', 'home_team']] = df[match_col].apply(
            lambda x: pd.Series(self.parse_match_string(x))
        )
        
        # Convert league names to sport_keys
        print("Converting league names to API sport keys...")
        df['sport_key'] = df[sport_key_col].apply(self.convert_league_to_sport_key)
        
        # Check for unmapped leagues
        unmapped = df[df['sport_key'].isna()][sport_key_col].unique()
        if len(unmapped) > 0:
            print(f"  ⚠ Warning: Could not map the following leagues to sport keys: {list(unmapped)}")
            print(f"    These rows will be skipped. Available mappings: {list(self.league_mapping.keys())}")
        
        # Filter out rows with unmapped leagues
        df_valid = df[df['sport_key'].notna()].copy()
        
        if len(df_valid) == 0:
            print("  ⚠ No valid sport keys found. Returning dataframe with empty event_id column.")
            df['event_id'] = None
            return df
        
        # Convert start times to UTC
        print("Converting times to UTC...")
        df_valid['start_time'] = pd.to_datetime(df_valid[start_time_col])
        df_valid['start_time_utc'] = df_valid['start_time'].apply(self.convert_to_utc)
        
        # Extract date for API calls (use UTC date)
        df_valid['date_utc'] = df_valid['start_time_utc'].dt.date
        
        # Group by date and sport to minimize API calls
        grouped = df_valid.groupby(['date_utc', 'sport_key'])
        
        print(f"\nProcessing {len(grouped)} unique date-sport combinations...")
        print(f"Total rows to match: {len(df_valid)}")
        print("-" * 70)
        
        # Initialize event_id column
        df['event_id'] = None
        
        matched_total = 0
        
        for (date_utc, sport_key), group_df in grouped:
            date_str = date_utc.isoformat() + 'T00:00:00Z'  # Add time component
            
            print(f"\n{sport_key} on {date_str}:")
            
            # Fetch events for this date and sport
            events = self.fetch_historical_events(sport_key, date_str)
            
            if not events:
                print(f"    ⚠ No events found")
                continue
            
            # Match each row in this group
            matched_count = 0
            for idx in group_df.index:
                row = df_valid.loc[idx]
                away_team = row['away_team']
                home_team = row['home_team']
                start_time_utc = row['start_time_utc']
                
                if pd.isna(away_team) or pd.isna(home_team):
                    continue
                
                event_id = self.match_event(away_team, home_team, start_time_utc, events)
                
                if event_id:
                    df.loc[idx, 'event_id'] = event_id
                    matched_count += 1
            
            matched_total += matched_count
            match_pct = (matched_count / len(group_df)) * 100
            print(f"    ✓ Matched {matched_count}/{len(group_df)} events ({match_pct:.1f}%)")
        
        print("\n" + "-" * 70)
        total_pct = (matched_total / len(df)) * 100
        print(f"✓ TOTAL: Matched {matched_total}/{len(df)} events ({total_pct:.1f}%)")
        
        return df


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("codebase/data/master_nc_avg_full.csv")
    
    print("Input DataFrame:")
    print(df.head())
    print(f"Total rows: {len(df)}")
    print(f"Leagues: {df['League'].unique()}")
    print("\n" + "=" * 70 + "\n")
    
    # Initialize matcher
    matcher = OddsAPIEventMatcher(
        api_key=THEODDS_API_KEY,
        timezone='America/New_York'  # EST/EDT timezone
    )
    
    # Add event IDs to dataframe
    df_with_ids = matcher.add_event_ids_to_df(
        df,
        sport_key_col='League',
        start_time_col='Start Time',
        match_col='Match'
    )
    
    print("\n" + "=" * 70)
    print("\nResulting DataFrame:")
    print(df_with_ids.head())
    print(f"\nRows with event_id: {df_with_ids['event_id'].notna().sum()}/{len(df_with_ids)}")
    
    # Save to CSV
    output_path = "codebase/data/clv.csv"
    df_with_ids.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")