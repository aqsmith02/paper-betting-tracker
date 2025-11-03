"""
test_fetch_odds.py

Unit tests for the fetch_odds.py module.

Author: Andrew Smith
"""

import unittest
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
        print(result)
        print(az_win)
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


if __name__ == "__main__":
    # Create test suite
    test_classes = [
        TestConvertToEasternTime,
        TestCreateBmDictList,
        TestProcessGame,
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
