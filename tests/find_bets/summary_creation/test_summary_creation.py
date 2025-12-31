"""
test_summary_creation.py

Unit tests for summary_creation.py module.

Author: Andrew Smith
"""

import unittest
import pandas as pd
import numpy as np
from src.find_bets.summary_creation import (
    create_average_edge_summary,
    create_zscore_summary,
    create_modified_zscore_summary,
    create_pinnacle_edge_summary,
    create_random_summary,
)


class TestCreateAverageEdgeSummary(unittest.TestCase):
    """Test cases for create_average_edge_summary function."""

    def setUp(self):
        """Set up test data before each test."""
        self.sample_data = {
            "Match": ["Team A vs Team B", "Team C vs Team D", "Team E vs Team F"],
            "League": ["Premier League", "La Liga", "Bundesliga"],
            "Team": ["Team A", "Team C", "Team E"],
            "Start Time": ["2025-11-01 15:00", "2025-11-01 18:00", "2025-11-02 20:00"],
            "Best Bookmaker": ["Bet365", "William Hill", "Betway"],
            "Best Odds": [2.5, 3.0, 1.8],
            "Expected Value": [0.15, 0.08, 0.12],
            "Fair Odds Avg": [2.3, 2.8, 1.7],
            "Result": ["Not Found", "Not Found", "Not Found"],
        }

    def test_create_summary_with_valid_data(self):
        """Test summary creation with valid data."""
        df = pd.DataFrame(self.sample_data)
        result = create_average_edge_summary(df)

        self.assertEqual(len(result), 3)
        self.assertIn("Match", result.columns)
        self.assertIn("League", result.columns)
        self.assertIn("Team", result.columns)
        self.assertIn("Avg Edge Book", result.columns)
        self.assertIn("Avg Edge Odds", result.columns)
        self.assertIn("Expected Value", result.columns)
        self.assertIn("Result", result.columns)

    def test_create_summary_filters_na_expected_value(self):
        """Test that rows with NaN Expected Value are filtered out."""
        data = self.sample_data.copy()
        data["Expected Value"] = [0.15, np.nan, 0.12]
        df = pd.DataFrame(data)
        result = create_average_edge_summary(df)

        self.assertEqual(len(result), 2)
        self.assertNotIn(np.nan, result["Expected Value"].values)

    def test_create_summary_filters_na_fair_odds(self):
        """Test that rows with NaN Fair Odds Avg are filtered out."""
        data = self.sample_data.copy()
        data["Fair Odds Avg"] = [2.3, np.nan, 1.7]
        df = pd.DataFrame(data)
        result = create_average_edge_summary(df)

        self.assertEqual(len(result), 2)

    def test_create_summary_with_empty_dataframe(self):
        """Test summary creation with empty DataFrame."""
        df = pd.DataFrame()
        result = create_average_edge_summary(df)

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)

    def test_create_summary_all_na_values(self):
        """Test summary creation when all Expected Values are NaN."""
        data = self.sample_data.copy()
        data["Expected Value"] = [np.nan, np.nan, np.nan]
        df = pd.DataFrame(data)
        result = create_average_edge_summary(df)

        self.assertEqual(len(result), 0)

    def test_column_mapping(self):
        """Test that columns are correctly mapped in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_average_edge_summary(df)

        self.assertEqual(result.iloc[0]["Avg Edge Book"], "Bet365")
        self.assertEqual(result.iloc[0]["Avg Edge Odds"], 2.5)
        self.assertEqual(result.iloc[0]["Match"], "Team A vs Team B")
        self.assertEqual(result.iloc[0]["Expected Value"], 0.15)

    def test_expected_value_preserved(self):
        """Test that Expected Value is correctly preserved in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_average_edge_summary(df)

        self.assertEqual(result.iloc[0]["Expected Value"], 0.15)
        self.assertEqual(result.iloc[1]["Expected Value"], 0.08)
        self.assertEqual(result.iloc[2]["Expected Value"], 0.12)


class TestCreateZscoreSummary(unittest.TestCase):
    """Test cases for create_zscore_summary function."""

    def setUp(self):
        """Set up test data before each test."""
        self.sample_data = {
            "Match": ["Team A vs Team B", "Team C vs Team D", "Team E vs Team F"],
            "League": ["Premier League", "La Liga", "Bundesliga"],
            "Team": ["Team A", "Team C", "Team E"],
            "Start Time": ["2025-11-01 15:00", "2025-11-01 18:00", "2025-11-02 20:00"],
            "Best Bookmaker": ["Bet365", "William Hill", "Betway"],
            "Best Odds": [2.5, 3.0, 1.8],
            "Z Score": [2.5, 3.2, 2.8],
            "Expected Value": [0.15, 0.08, 0.12],
            "Result": ["Win", "Loss", "Win"],
        }

    def test_create_zscore_summary_with_valid_data(self):
        """Test Z-score summary creation with valid data."""
        df = pd.DataFrame(self.sample_data)
        result = create_zscore_summary(df)

        self.assertEqual(len(result), 3)
        self.assertIn("Z Score", result.columns)
        self.assertIn("Outlier Book", result.columns)
        self.assertIn("Outlier Odds", result.columns)
        self.assertIn("Expected Value", result.columns)

    def test_filters_na_zscore(self):
        """Test that rows with NaN Z Score are filtered out."""
        data = self.sample_data.copy()
        data["Z Score"] = [2.5, np.nan, 2.8]
        df = pd.DataFrame(data)
        result = create_zscore_summary(df)

        self.assertEqual(len(result), 2)

    def test_filters_na_expected_value(self):
        """Test that rows with NaN Expected Value are filtered out."""
        data = self.sample_data.copy()
        data["Expected Value"] = [0.15, np.nan, 0.12]
        df = pd.DataFrame(data)
        result = create_zscore_summary(df)

        self.assertEqual(len(result), 2)

    def test_result_default_value(self):
        """Test that Result defaults to 'Not Found' when missing."""
        data = self.sample_data.copy()
        del data["Result"]
        df = pd.DataFrame(data)
        result = create_zscore_summary(df)

        self.assertTrue(all(result["Result"] == "Not Found"))

    def test_with_empty_dataframe(self):
        """Test Z-score summary with empty DataFrame."""
        df = pd.DataFrame()
        result = create_zscore_summary(df)

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_value_in_output(self):
        """Test that Expected Value is included in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_zscore_summary(df)

        self.assertEqual(result.iloc[0]["Expected Value"], 0.15)
        self.assertEqual(result.iloc[1]["Expected Value"], 0.08)

    def test_all_columns_present(self):
        """Test that all expected columns are present in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_zscore_summary(df)

        expected_cols = [
            "Match", "League", "Team", "Start Time",
            "Outlier Book", "Outlier Odds", "Z Score",
            "Expected Value", "Result"
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)


class TestCreateModifiedZscoreSummary(unittest.TestCase):
    """Test cases for create_modified_zscore_summary function."""

    def setUp(self):
        """Set up test data before each test."""
        self.sample_data = {
            "Match": ["Team A vs Team B", "Team C vs Team D"],
            "League": ["Premier League", "La Liga"],
            "Team": ["Team A", "Team C"],
            "Start Time": ["2025-11-01 15:00", "2025-11-01 18:00"],
            "Best Bookmaker": ["Bet365", "William Hill"],
            "Best Odds": [2.5, 3.0],
            "Modified Z Score": [3.5, 4.2],
            "Expected Value": [0.15, 0.08],
            "Result": ["Win", "Loss"],
        }

    def test_create_modified_zscore_summary_valid_data(self):
        """Test Modified Z-score summary creation with valid data."""
        df = pd.DataFrame(self.sample_data)
        result = create_modified_zscore_summary(df)

        self.assertEqual(len(result), 2)
        self.assertIn("Modified Z Score", result.columns)
        self.assertIn("Outlier Book", result.columns)
        self.assertIn("Expected Value", result.columns)

    def test_filters_na_modified_zscore(self):
        """Test that rows with NaN Modified Z Score are filtered out."""
        data = self.sample_data.copy()
        data["Modified Z Score"] = [3.5, np.nan]
        df = pd.DataFrame(data)
        result = create_modified_zscore_summary(df)

        self.assertEqual(len(result), 1)

    def test_filters_na_expected_value(self):
        """Test that rows with NaN Expected Value are filtered out."""
        data = self.sample_data.copy()
        data["Expected Value"] = [np.nan, 0.08]
        df = pd.DataFrame(data)
        result = create_modified_zscore_summary(df)

        self.assertEqual(len(result), 1)

    def test_result_default_value(self):
        """Test that Result defaults to 'Not Found' when missing."""
        data = self.sample_data.copy()
        del data["Result"]
        df = pd.DataFrame(data)
        result = create_modified_zscore_summary(df)

        self.assertTrue(all(result["Result"] == "Not Found"))

    def test_column_values(self):
        """Test that column values are correctly mapped."""
        df = pd.DataFrame(self.sample_data)
        result = create_modified_zscore_summary(df)

        self.assertEqual(result.iloc[0]["Modified Z Score"], 3.5)
        self.assertEqual(result.iloc[0]["Outlier Book"], "Bet365")
        self.assertEqual(result.iloc[0]["Outlier Odds"], 2.5)
        self.assertEqual(result.iloc[0]["Expected Value"], 0.15)

    def test_empty_dataframe(self):
        """Test Modified Z-score summary with empty DataFrame."""
        df = pd.DataFrame()
        result = create_modified_zscore_summary(df)

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)


class TestCreatePinnacleEdgeSummary(unittest.TestCase):
    """Test cases for create_pinnacle_edge_summary function."""

    def setUp(self):
        """Set up test data before each test."""
        self.sample_data = {
            "Match": ["Team A vs Team B", "Team C vs Team D"],
            "League": ["Premier League", "La Liga"],
            "Team": ["Team A", "Team C"],
            "Start Time": ["2025-11-01 15:00", "2025-11-01 18:00"],
            "Best Bookmaker": ["Bet365", "William Hill"],
            "Best Odds": [2.5, 3.0],
            "Pinnacle Fair Odds": [2.3, 2.8],
            "Expected Value": [0.18, 0.10],
            "Vigfree Pinnacle": [0.43, 0.36],
            "Result": ["Win", "Loss"],
        }

    def test_create_pinnacle_edge_summary_valid_data(self):
        """Test Pinnacle edge summary creation with valid data."""
        df = pd.DataFrame(self.sample_data)
        result = create_pinnacle_edge_summary(df)

        self.assertEqual(len(result), 2)
        self.assertIn("Pinnacle Edge Book", result.columns)
        self.assertIn("Pinnacle Edge Odds", result.columns)
        self.assertIn("Expected Value", result.columns)
        self.assertIn("Pinnacle Fair Odds", result.columns)

    def test_missing_vigfree_pinnacle_column(self):
        """Test behavior when Vigfree Pinnacle column is missing."""
        data = self.sample_data.copy()
        del data["Vigfree Pinnacle"]
        df = pd.DataFrame(data)
        result = create_pinnacle_edge_summary(df)

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)

    def test_filters_na_pinnacle_fair_odds(self):
        """Test that rows with NaN Pinnacle Fair Odds are filtered out."""
        data = self.sample_data.copy()
        data["Pinnacle Fair Odds"] = [np.nan, 2.8]
        df = pd.DataFrame(data)
        result = create_pinnacle_edge_summary(df)

        self.assertEqual(len(result), 1)

    def test_filters_na_expected_value(self):
        """Test that rows with NaN Expected Value are filtered out."""
        data = self.sample_data.copy()
        data["Expected Value"] = [0.18, np.nan]
        df = pd.DataFrame(data)
        result = create_pinnacle_edge_summary(df)

        self.assertEqual(len(result), 1)

    def test_result_default_value(self):
        """Test that Result defaults to 'Not Found' when missing."""
        data = self.sample_data.copy()
        del data["Result"]
        df = pd.DataFrame(data)
        result = create_pinnacle_edge_summary(df)

        self.assertTrue(all(result["Result"] == "Not Found"))

    def test_empty_dataframe(self):
        """Test Pinnacle edge summary with empty DataFrame."""
        df = pd.DataFrame()
        result = create_pinnacle_edge_summary(df)

        self.assertEqual(len(result), 0)

    def test_expected_value_in_output(self):
        """Test that Expected Value is correctly included in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_pinnacle_edge_summary(df)

        self.assertEqual(result.iloc[0]["Expected Value"], 0.18)
        self.assertEqual(result.iloc[1]["Expected Value"], 0.10)

    def test_all_columns_present(self):
        """Test that all expected columns are present in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_pinnacle_edge_summary(df)

        expected_cols = [
            "Match", "League", "Team", "Start Time",
            "Pinnacle Edge Book", "Pinnacle Edge Odds",
            "Expected Value", "Pinnacle Fair Odds", "Result"
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)


class TestCreateRandomSummary(unittest.TestCase):
    """Test cases for create_random_summary function."""

    def setUp(self):
        """Set up test data before each test."""
        self.sample_data = {
            "Match": ["Team A vs Team B", "Team C vs Team D", "Team E vs Team F"],
            "League": ["Premier League", "La Liga", "Bundesliga"],
            "Team": ["Team A", "Team C", "Team E"],
            "Start Time": ["2025-11-01 15:00", "2025-11-01 18:00", "2025-11-02 20:00"],
            "Best Bookmaker": ["Bet365", "William Hill", "Betway"],
            "Best Odds": [2.5, 3.0, 1.8],
            "Random Placed Bet": [1, 0, 1],
            "Result": ["Win", "Loss", "Win"],
        }

    def test_create_random_summary_valid_data(self):
        """Test random summary creation with valid data."""
        df = pd.DataFrame(self.sample_data)
        result = create_random_summary(df)

        self.assertEqual(len(result), 2)
        self.assertIn("Random Bet Book", result.columns)
        self.assertIn("Random Bet Odds", result.columns)

    def test_filters_zero_random_bets(self):
        """Test that rows with Random Placed Bet = 0 are filtered out."""
        df = pd.DataFrame(self.sample_data)
        result = create_random_summary(df)

        self.assertTrue(all(result["Random Bet Book"].isin(["Bet365", "Betway"])))
        self.assertNotIn("William Hill", result["Random Bet Book"].values)

    def test_all_zeros(self):
        """Test when all Random Placed Bet values are 0."""
        data = self.sample_data.copy()
        data["Random Placed Bet"] = [0, 0, 0]
        df = pd.DataFrame(data)
        result = create_random_summary(df)

        self.assertEqual(len(result), 0)

    def test_all_ones(self):
        """Test when all Random Placed Bet values are 1."""
        data = self.sample_data.copy()
        data["Random Placed Bet"] = [1, 1, 1]
        df = pd.DataFrame(data)
        result = create_random_summary(df)

        self.assertEqual(len(result), 3)

    def test_result_default_value(self):
        """Test that Result defaults to 'Not Found' when missing."""
        data = self.sample_data.copy()
        del data["Result"]
        df = pd.DataFrame(data)
        result = create_random_summary(df)

        self.assertTrue(all(result["Result"] == "Not Found"))

    def test_empty_dataframe(self):
        """Test random summary with empty DataFrame."""
        df = pd.DataFrame()
        result = create_random_summary(df)

        self.assertEqual(len(result), 0)

    def test_column_mapping(self):
        """Test that columns are correctly mapped in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_random_summary(df)

        self.assertEqual(result.iloc[0]["Random Bet Book"], "Bet365")
        self.assertEqual(result.iloc[0]["Random Bet Odds"], 2.5)
        self.assertEqual(result.iloc[0]["Result"], "Win")

    def test_all_columns_present(self):
        """Test that all expected columns are present in output."""
        df = pd.DataFrame(self.sample_data)
        result = create_random_summary(df)

        expected_cols = [
            "Match", "League", "Team", "Start Time",
            "Random Bet Book", "Random Bet Odds", "Result"
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)