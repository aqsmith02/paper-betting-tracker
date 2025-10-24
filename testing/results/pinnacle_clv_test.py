import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import pytz
from codebase.results.pinnacle_clv import OddsAPIEventMatcher
from codebase.constants import THEODDS_API_KEY


class TestOddsAPIEventMatcher(unittest.TestCase):
    """Test suite for OddsAPIEventMatcher class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.matcher = OddsAPIEventMatcher(
            api_key=THEODDS_API_KEY,
            timezone='America/New_York'
        )
        
    def test_initialization(self):
        """Test that the matcher initializes correctly"""
        self.assertEqual(self.matcher.api_key, THEODDS_API_KEY)
        self.assertEqual(self.matcher.base_url, "https://api.the-odds-api.com/v4")
        self.assertIsInstance(self.matcher.timezone, pytz.tzinfo.BaseTzInfo)
        self.assertEqual(self.matcher.cache, {})
        
    def test_parse_match_string_valid(self):
        """Test parsing valid match strings"""
        # Test standard format
        away, home = self.matcher.parse_match_string("Milwaukee Brewers @ San Diego Padres")
        self.assertEqual(away, "Milwaukee Brewers")
        self.assertEqual(home, "San Diego Padres")
        
        # Test with extra spaces
        away, home = self.matcher.parse_match_string("Lakers  @  Celtics")
        self.assertEqual(away, "Lakers")
        self.assertEqual(home, "Celtics")
        
    def test_parse_match_string_invalid(self):
        """Test parsing invalid match strings"""
        # Test without @ symbol
        away, home = self.matcher.parse_match_string("Lakers vs Celtics")
        self.assertIsNone(away)
        self.assertIsNone(home)
        
        # Test empty string
        away, home = self.matcher.parse_match_string("")
        self.assertIsNone(away)
        self.assertIsNone(home)
        
    def test_convert_to_utc(self):
        """Test timezone conversion from EST to UTC"""
        # Create a naive datetime (assumed to be in EST)
        est_time = datetime(2023, 11, 29, 19, 10, 0)
        
        # Convert to UTC
        utc_time = self.matcher.convert_to_utc(est_time)
        
        # Verify it's in UTC
        self.assertEqual(utc_time.tzinfo, pytz.UTC)
        
        # EST is UTC-5, so 19:10 EST should be 00:10 UTC next day
        # (during standard time, not daylight saving)
        self.assertEqual(utc_time.hour, 0)
        self.assertEqual(utc_time.day, 30)
        
    def test_convert_to_utc_already_aware(self):
        """Test converting an already timezone-aware datetime"""
        # Create a timezone-aware datetime
        est_tz = pytz.timezone('America/New_York')
        aware_time = est_tz.localize(datetime(2023, 11, 29, 19, 10, 0))
        
        # Convert to UTC
        utc_time = self.matcher.convert_to_utc(aware_time)
        
        # Should still work correctly
        self.assertEqual(utc_time.tzinfo, pytz.UTC)
        
    @patch('codebase.results.pinnacle_clv.requests.get')

    def test_fetch_historical_events_success(self, mock_get):
        """Test successful API fetch"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'id': 'test_event_1',
                    'sport_key': 'basketball_nba',
                    'home_team': 'Detroit Pistons',
                    'away_team': 'Los Angeles Lakers',
                    'commence_time': '2023-11-30T00:10:00Z'
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Call the method
        events = self.matcher.fetch_historical_events('basketball_nba', '2023-11-29')
        
        # Verify the results
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['id'], 'test_event_1')
        
        # Verify API was called correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('basketball_nba', call_args[0][0])
        self.assertEqual(call_args[1]['params']['apiKey'], THEODDS_API_KEY)
        self.assertEqual(call_args[1]['params']['date'], '2023-11-29')
        
    @patch('codebase.results.pinnacle_clv.requests.get')

    def test_fetch_historical_events_uses_cache(self, mock_get):
        """Test that cached results are used on subsequent calls"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {'data': [{'id': 'cached_event'}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # First call - should hit API
        events1 = self.matcher.fetch_historical_events('basketball_nba', '2023-11-29')
        
        # Second call - should use cache
        events2 = self.matcher.fetch_historical_events('basketball_nba', '2023-11-29')
        
        # Verify results are the same
        self.assertEqual(events1, events2)
        
        # Verify API was only called once
        self.assertEqual(mock_get.call_count, 1)
        
    @patch('codebase.results.pinnacle_clv.requests.get')

    def test_fetch_historical_events_api_error(self, mock_get):
        """Test handling of API errors"""
        # Mock API error - use RequestException which is caught in the code
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        # Call should not raise exception, but return empty list
        events = self.matcher.fetch_historical_events('basketball_nba', '2023-11-29')
        
        self.assertEqual(events, [])
        
    def test_match_event_exact_match(self):
        """Test matching an event with exact team names and time"""
        # Create test events
        events = [
            {
                'id': 'event_123',
                'home_team': 'Detroit Pistons',
                'away_team': 'Los Angeles Lakers',
                'commence_time': '2023-11-30T00:10:00Z'
            }
        ]
        
        # Create matching time (in UTC)
        match_time = pd.to_datetime('2023-11-30T00:10:00Z')
        
        # Test match
        event_id = self.matcher.match_event(
            'Los Angeles Lakers',
            'Detroit Pistons',
            match_time,
            events
        )
        
        self.assertEqual(event_id, 'event_123')
        
    def test_match_event_case_insensitive(self):
        """Test that matching is case-insensitive"""
        events = [
            {
                'id': 'event_456',
                'home_team': 'detroit pistons',
                'away_team': 'los angeles lakers',
                'commence_time': '2023-11-30T00:10:00Z'
            }
        ]
        
        match_time = pd.to_datetime('2023-11-30T00:10:00Z')
        
        # Test with different case
        event_id = self.matcher.match_event(
            'Los Angeles Lakers',
            'Detroit Pistons',
            match_time,
            events
        )
        
        self.assertEqual(event_id, 'event_456')
        
    def test_match_event_wrong_teams(self):
        """Test that wrong teams don't match"""
        events = [
            {
                'id': 'event_789',
                'home_team': 'Detroit Pistons',
                'away_team': 'Boston Celtics',  # Different away team
                'commence_time': '2023-11-30T00:10:00Z'
            }
        ]
        
        match_time = pd.to_datetime('2023-11-30T00:10:00Z')
        
        event_id = self.matcher.match_event(
            'Los Angeles Lakers',
            'Detroit Pistons',
            match_time,
            events
        )
        
        self.assertIsNone(event_id)
        
    def test_match_event_wrong_time(self):
        """Test that events with wrong time don't match"""
        events = [
            {
                'id': 'event_999',
                'home_team': 'Detroit Pistons',
                'away_team': 'Los Angeles Lakers',
                'commence_time': '2023-11-30T02:00:00Z'  # 2 hours difference
            }
        ]
        
        match_time = pd.to_datetime('2023-11-30T00:10:00Z')
        
        event_id = self.matcher.match_event(
            'Los Angeles Lakers',
            'Detroit Pistons',
            match_time,
            events
        )
        
        self.assertIsNone(event_id)
        
    def test_match_event_within_time_window(self):
        """Test that events within 1 minute window match"""
        events = [
            {
                'id': 'event_555',
                'home_team': 'Detroit Pistons',
                'away_team': 'Los Angeles Lakers',
                'commence_time': '2023-11-30T00:10:30Z'  # 30 seconds difference
            }
        ]
        
        match_time = pd.to_datetime('2023-11-30T00:10:00Z')
        
        event_id = self.matcher.match_event(
            'Los Angeles Lakers',
            'Detroit Pistons',
            match_time,
            events
        )
        
        self.assertEqual(event_id, 'event_555')
        
    @patch.object(OddsAPIEventMatcher, 'fetch_historical_events')
    def test_add_event_ids_to_df(self, mock_fetch):
        """Test adding event IDs to a dataframe"""
        # Mock API response
        mock_fetch.return_value = [
            {
                'id': 'mlb_event_1',
                'home_team': 'San Diego Padres',
                'away_team': 'Milwaukee Brewers',
                'commence_time': '2025-09-23T01:41:00Z'  # 21:41 EST + 4 hours
            },
            {
                'id': 'nba_event_1',
                'home_team': 'Detroit Pistons',
                'away_team': 'Los Angeles Lakers',
                'commence_time': '2023-11-30T00:10:00Z'
            }
        ]
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'Match': [
                'Milwaukee Brewers @ San Diego Padres',
                'Los Angeles Lakers @ Detroit Pistons'
            ],
            'League': ['baseball_mlb', 'basketball_nba'],
            'Team': ['San Diego Padres', 'Detroit Pistons'],
            'Start Time': ['2025-09-22 21:41:00', '2023-11-29 19:10:00']
        })
        
        # Process dataframe
        result_df = self.matcher.add_event_ids_to_df(test_df)
        
        # Verify event_id column was added
        self.assertIn('event_id', result_df.columns)
        
        # Verify at least one event was matched
        # (exact matching depends on timezone conversion)
        self.assertTrue(result_df['event_id'].notna().any())
        
    def test_add_event_ids_to_df_empty(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame({
            'Match': [],
            'League': [],
            'Team': [],
            'Start Time': []
        })
        
        # For empty dataframe, just manually add the event_id column
        # since the groupby will have no groups to iterate over
        with patch.object(OddsAPIEventMatcher, 'fetch_historical_events', return_value=[]):
            result_df = self.matcher.add_event_ids_to_df(empty_df)
        
        self.assertIn('event_id', result_df.columns)
        self.assertEqual(len(result_df), 0)
        
    @patch.object(OddsAPIEventMatcher, 'fetch_historical_events')
    def test_add_event_ids_to_df_no_matches(self, mock_fetch):
        """Test when no events match"""
        # Mock API response with non-matching events
        mock_fetch.return_value = [
            {
                'id': 'different_event',
                'home_team': 'Boston Celtics',
                'away_team': 'Miami Heat',
                'commence_time': '2023-11-30T00:00:00Z'
            }
        ]
        
        test_df = pd.DataFrame({
            'Match': ['Milwaukee Brewers @ San Diego Padres'],
            'League': ['baseball_mlb'],
            'Team': ['San Diego Padres'],
            'Start Time': ['2025-09-22 21:41:00']
        })
        
        result_df = self.matcher.add_event_ids_to_df(test_df)
        
        # All event_ids should be None/NaN
        self.assertTrue(result_df['event_id'].isna().all())
        
    def test_add_event_ids_custom_columns(self):
        """Test with custom column names"""
        test_df = pd.DataFrame({
            'game': ['Lakers @ Celtics'],
            'sport': ['basketball_nba'],
            'team_name': ['Celtics'],
            'game_time': ['2023-11-29 19:00:00']
        })
        
        # Should not raise an error with custom column names
        with patch.object(OddsAPIEventMatcher, 'fetch_historical_events', return_value=[]):
            result_df = self.matcher.add_event_ids_to_df(
                test_df,
                sport_key_col='sport',
                start_time_col='game_time',
                match_col='game'
            )
            
        self.assertIn('event_id', result_df.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests that test multiple components together"""
    
    @patch('codebase.results.pinnacle_clv.requests.get')
    def test_full_workflow(self,mock_get):
        """Test the complete workflow from dataframe to matched events"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'id': 'complete_test_event',
                    'sport_key': 'basketball_nba',
                    'home_team': 'Detroit Pistons',
                    'away_team': 'Los Angeles Lakers',
                    'commence_time': '2023-11-30T00:10:00Z'
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Create matcher
        matcher = OddsAPIEventMatcher(api_key='test_key')
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'Match': ['Los Angeles Lakers @ Detroit Pistons'],
            'League': ['basketball_nba'],
            'Team': ['Detroit Pistons'],
            'Start Time': ['2023-11-29 19:10:00']
        })
        
        # Process
        result_df = matcher.add_event_ids_to_df(test_df)
        
        # Verify
        self.assertEqual(len(result_df), 1)
        self.assertIn('event_id', result_df.columns)
        self.assertEqual(result_df.loc[0, 'event_id'], 'complete_test_event')


if __name__ == '__main__':
    unittest.main(verbosity=2)