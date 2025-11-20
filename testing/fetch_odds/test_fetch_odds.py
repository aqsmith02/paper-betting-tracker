"""
test_fetch_odds.py

Unit tests for the fetch_odds.py module.

Author: Andrew Smith
"""

import unittest
from unittest.mock import patch, Mock
from .configs import (
    TEST_GAME_DATA,
    TEST_GAME_DATA_NO_BM,
    TEST_GAME_DATA_MULT_MARKETS,
)

# Import the module under test
from codebase.fetch_odds.fetch_odds import (
    _convert_to_eastern_time,
    _process_game,
    _create_bm_dict_list,
    fetch_odds,
)


class TestConvertToEasternTime(unittest.TestCase):
    """Test cases for _convert_to_eastern_time function."""

    def test_convert_utc_to_eastern_standard_time(self):
        """Test conversion during Eastern Standard Time (EST)."""
        # January 15, 2025 12:00 UTC should be 7:00 AM EST
        utc_time = "2025-01-15T12:00:00Z"
        result = _convert_to_eastern_time(utc_time)
        self.assertIn("2025-01-15", result)
        self.assertIn("07:00", result)

    def test_convert_utc_to_eastern_daylight_time(self):
        """Test conversion during Eastern Daylight Time (EDT)."""
        # July 15, 2025 12:00 UTC should be 8:00 AM EDT
        utc_time = "2025-07-15T12:00:00Z"
        result = _convert_to_eastern_time(utc_time)
        self.assertIn("2025-07-15", result)
        self.assertIn("08:00", result)

    def test_convert_edge_case_midnight(self):
        """Test conversion of midnight UTC."""
        utc_time = "2025-01-15T00:00:00Z"
        result = _convert_to_eastern_time(utc_time)
        # Should be previous day in Eastern time
        self.assertIn("2025-01-14", result)
        self.assertIn("19:00", result)


class TestCreateBmDictList(unittest.TestCase):
    """Test cases for _create_bm_dict_list function."""

    def setUp(self):
        """Set up test data."""
        self.test_data = TEST_GAME_DATA
        self.test_data_no_bm = TEST_GAME_DATA_NO_BM
        self.test_data_mult_markets = TEST_GAME_DATA_MULT_MARKETS

    def test_create_bm_dict_list_normal_case(self):
        """Test normal case with multiple bookmakers."""
        result = _create_bm_dict_list(self.test_data)

        self.assertEqual(len(result), 2)
        self.assertIn("FanDuel", result[0])
        self.assertIn("888sport", result[1])

        # Check FanDuel odds
        fd_odds = result[0]["FanDuel"]
        self.assertEqual(fd_odds["Arizona Diamondbacks"], 1.67)
        self.assertEqual(fd_odds["Los Angeles Dodgers"], 2.18)

        # Check 888sport odds
        eight_odds = result[1]["888sport"]
        self.assertEqual(eight_odds["Arizona Diamondbacks"], 1.95)
        self.assertEqual(eight_odds["Los Angeles Dodgers"], 1.73)

    def test_create_bm_dict_list_no_bookmakers(self):
        """Test case with no bookmakers."""
        result = _create_bm_dict_list(self.test_data_no_bm)
        self.assertEqual(result, [])

    def test_create_bm_dict_list_multiple_markets(self):
        """Test case with multiple markets."""
        result = _create_bm_dict_list(self.test_data_mult_markets)
        smarkets1_odds = result[0]["Smarkets1"]
        self.assertEqual(smarkets1_odds["Kansas City Royals"], 1.01)
        self.assertEqual(smarkets1_odds["Los Angeles Angels"], 17.99)

        smarkets2_odds = result[1]["Smarkets2"]
        self.assertEqual(smarkets2_odds["Kansas City Royals"], 1.01)
        self.assertEqual(smarkets2_odds["Los Angeles Angels"], 17.99)


class TestProcessGame(unittest.TestCase):
    """Test cases for _process_game function."""

    def setUp(self):
        """Set up test data."""
        self.test_data = TEST_GAME_DATA
        self.test_data_no_bm = TEST_GAME_DATA_NO_BM
        self.test_data_mult_markets = TEST_GAME_DATA_MULT_MARKETS

    def test_process_game_normal_case(self):
        """Test normal game processing."""
        result = _process_game(self.test_data)
        az_win = result[0]
        la_win = result[1]

        # Should have 2 rows (one for each team)
        self.assertEqual(len(result), 2)

        self.assertEqual(az_win["match"], "Kansas City Royals @ Los Angeles Angels")
        self.assertEqual(az_win["league"], "MLB")
        self.assertEqual(az_win["start_time"], "2025-09-23 21:39:00")
        self.assertEqual(az_win["team"], "Arizona Diamondbacks")
        self.assertEqual(az_win["FanDuel"], 1.67)
        self.assertEqual(az_win["888sport"], 1.95)

        self.assertEqual(la_win["match"], "Kansas City Royals @ Los Angeles Angels")
        self.assertEqual(la_win["league"], "MLB")
        self.assertEqual(la_win["start_time"], "2025-09-23 21:39:00")
        self.assertEqual(la_win["team"], "Los Angeles Dodgers")
        self.assertEqual(la_win["FanDuel"], 2.18)
        self.assertEqual(la_win["888sport"], 1.73)

    def test_process_game_no_bookmakers(self):
        """Test game processing with no bookmakers."""
        result = _process_game(self.test_data_no_bm)
        self.assertEqual(result, [])


class TestFetchOddsAPIFailures(unittest.TestCase):
    """Test cases for fetch_odds function with API failures."""

    @patch('codebase.fetch_odds.fetch_odds.requests.get')
    def test_fetch_odds_invalid_api_key(self, mock_get):
        """Test fetch_odds when API key is invalid (401 Unauthorized)."""
        # Mock a 401 Unauthorized response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.headers.get.return_value = "0"
        mock_get.return_value = mock_response

        result = fetch_odds()

        # Should return empty DataFrame
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)

    @patch('codebase.fetch_odds.fetch_odds.requests.get')
    def test_fetch_odds_api_key_quota_exceeded(self, mock_get):
        """Test fetch_odds when API key quota is exceeded (429 Too Many Requests)."""
        # Mock a 429 Too Many Requests response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Request quota exceeded"
        mock_response.headers.get.return_value = "0"
        mock_get.return_value = mock_response

        result = fetch_odds()

        # Should return empty DataFrame
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)

    @patch('codebase.fetch_odds.fetch_odds.requests.get')
    def test_fetch_odds_forbidden_access(self, mock_get):
        """Test fetch_odds when API returns 403 Forbidden."""
        # Mock a 403 Forbidden response
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden - API key does not have access to this resource"
        mock_response.headers.get.return_value = "0"
        mock_get.return_value = mock_response

        result = fetch_odds()

        # Should return empty DataFrame
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)

    @patch('codebase.fetch_odds.fetch_odds.requests.get')
    def test_fetch_odds_server_error(self, mock_get):
        """Test fetch_odds when API returns 500 Internal Server Error."""
        # Mock a 500 Internal Server Error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.headers.get.return_value = "100"
        mock_get.return_value = mock_response

        result = fetch_odds()

        # Should return empty DataFrame
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)

    @patch('codebase.fetch_odds.fetch_odds.requests.get')
    def test_fetch_odds_network_timeout(self, mock_get):
        """Test fetch_odds when network request times out."""
        # Mock a timeout exception
        mock_get.side_effect = Exception("Connection timeout")

        result = fetch_odds()

        # Should return empty DataFrame
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)

    @patch('codebase.fetch_odds.fetch_odds.requests.get')
    def test_fetch_odds_successful_with_valid_key(self, mock_get):
        """Test fetch_odds with valid API key returning game data."""
        # Mock a successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [TEST_GAME_DATA]
        mock_response.headers.get.side_effect = lambda x: {
            'x-requests-remaining': '499',
            'x-requests-used': '1'
        }.get(x)
        mock_get.return_value = mock_response

        result = fetch_odds()

        # Should return non-empty DataFrame
        self.assertFalse(result.empty)
        self.assertGreater(len(result), 0)
        # Check that expected columns exist
        self.assertIn("match", result.columns)
        self.assertIn("league", result.columns)
        self.assertIn("team", result.columns)


if __name__ == "__main__":
    # Create test suite
    test_classes = [
        TestConvertToEasternTime,
        TestCreateBmDictList,
        TestProcessGame,
        TestFetchOddsAPIFailures,
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )