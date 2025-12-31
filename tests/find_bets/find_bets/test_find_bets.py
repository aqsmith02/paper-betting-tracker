"""
Unit tests for find_bets.py

Tests the main betting analysis pipeline including strategy execution,
data processing, and file management integration.

Author: Andrew Smith
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the parent directory to the path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.find_bets.find_bets import (
    BettingStrategy,
    run_betting_strategy,
    main,
)


class TestBettingStrategy(unittest.TestCase):
    """Test the BettingStrategy dataclass."""

    def test_betting_strategy_creation(self):
        """Test creating a BettingStrategy instance."""
        mock_func = Mock()
        strategy = BettingStrategy(
            name="Test Strategy",
            nc_summary_file="test_nc_summary.csv",
            nc_full_file="test_nc_full.csv",
            score_column="Test Score",
            summary_func=mock_func,
            analysis_func=mock_func,
        )

        self.assertEqual(strategy.name, "Test Strategy")
        self.assertEqual(strategy.nc_summary_file, "test_nc_summary.csv")
        self.assertEqual(strategy.nc_full_file, "test_nc_full.csv")
        self.assertEqual(strategy.score_column, "Test Score")
        self.assertEqual(strategy.summary_func, mock_func)
        self.assertEqual(strategy.analysis_func, mock_func)


class TestRunBettingStrategy(unittest.TestCase):
    """Test the run_betting_strategy function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock NC data
        self.nc_df = pd.DataFrame(
            {
                "Match": ["Team E vs Team F"],
                "Start Time": ["2025-11-04T21:00:00Z"],
                "Best Odds": [2.5],
            }
        )

        # Create mock analysis result
        self.nc_analysis_result = pd.DataFrame(
            {
                "Match": ["Team E vs Team F"],
                "Start Time": ["2025-11-04T21:00:00Z"],
                "Outcome": ["Team E"],
                "Best Odds": [2.5],
                "Score": [4.8],
            }
        )

        # Create mock summary
        self.nc_summary = pd.DataFrame(
            {
                "Match": ["Team E vs Team F"],
                "Start Time": ["2025-11-04T21:00:00Z"],
                "Outcome": ["Team E"],
                "Best Odds": [2.5],
                "Score": [4.8],
            }
        )

        # Create mock functions
        self.mock_analysis_func = Mock(return_value=self.nc_analysis_result)
        self.mock_summary_func = Mock(return_value=self.nc_summary)

        # Create mock file manager
        self.mock_file_manager = Mock()
        self.mock_file_manager.save_best_bets_only.return_value = self.nc_summary
        self.mock_file_manager.save_full_betting_data.return_value = None

        # Create strategy
        self.strategy = BettingStrategy(
            name="Test Strategy",
            nc_summary_file="test_nc_summary.csv",
            nc_full_file="test_nc_full.csv",
            score_column="Score",
            summary_func=self.mock_summary_func,
            analysis_func=self.mock_analysis_func,
        )

    def test_run_betting_strategy_success(self):
        """Test successful execution of betting strategy."""
        run_betting_strategy(
            self.strategy, self.nc_df, self.mock_file_manager
        )

        # Verify analysis function was called
        self.mock_analysis_func.assert_called_once_with(self.nc_df)
        
        # Verify summary function was called
        self.mock_summary_func.assert_called_once_with(self.nc_analysis_result)

        # Verify file manager methods were called correctly
        self.mock_file_manager.save_best_bets_only.assert_called_once()
        self.mock_file_manager.save_full_betting_data.assert_called_once()

    def test_run_betting_strategy_empty_nc_summary(self):
        """Test when NC summary is empty."""
        # Mock empty summary for NC
        self.mock_summary_func = Mock(return_value=pd.DataFrame())
        self.mock_analysis_func = Mock(return_value=self.nc_analysis_result)
        
        strategy = BettingStrategy(
            name="Test Strategy",
            nc_summary_file="test_nc_summary.csv",
            nc_full_file="test_nc_full.csv",
            score_column="Score",
            summary_func=self.mock_summary_func,
            analysis_func=self.mock_analysis_func,
        )

        run_betting_strategy(strategy, self.nc_df, self.mock_file_manager)
        
        # File manager should not be called when summary is empty
        self.mock_file_manager.save_best_bets_only.assert_not_called()
        self.mock_file_manager.save_full_betting_data.assert_not_called()

    def test_run_betting_strategy_nc_exception(self):
        """Test handling of exception during NC analysis."""
        # Mock exception for NC analysis
        self.mock_analysis_func = Mock(side_effect=Exception("Test error"))
        
        strategy = BettingStrategy(
            name="Test Strategy",
            nc_summary_file="test_nc_summary.csv",
            nc_full_file="test_nc_full.csv",
            score_column="Score",
            summary_func=self.mock_summary_func,
            analysis_func=self.mock_analysis_func,
        )

        # Should not raise an error - catches exception internally
        run_betting_strategy(strategy, self.nc_df, self.mock_file_manager)
        
        # File manager should not be called due to exception
        self.mock_file_manager.save_best_bets_only.assert_not_called()


class TestMain(unittest.TestCase):
    """Test the main function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_raw_odds = pd.DataFrame(
            {
                "sport_title": ["Basketball", "Basketball"],
                "commence_time": ["2025-11-04T19:00:00Z", "2025-11-04T20:00:00Z"],
                "home_team": ["Team A", "Team C"],
                "away_team": ["Team B", "Team D"],
            }
        )

        self.mock_processed_odds = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team C vs Team D"],
                "Start Time": ["2025-11-04T19:00:00Z", "2025-11-04T20:00:00Z"],
            }
        )

        self.mock_vigfree_data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team C vs Team D"],
                "Start Time": ["2025-11-04T19:00:00Z", "2025-11-04T20:00:00Z"],
                "Vigfree Prob": [0.45, 0.52],
            }
        )

    @patch("codebase.find_bets.find_bets.BetFileManager")
    @patch("codebase.find_bets.find_bets.calculate_vigfree_probabilities")
    @patch("codebase.find_bets.find_bets.process_target_odds_data")
    @patch("codebase.find_bets.find_bets.fetch_odds")
    @patch("codebase.find_bets.find_bets.run_betting_strategy")
    def test_main_success(
        self,
        mock_run_strategy,
        mock_fetch_odds,
        mock_process_target,
        mock_calc_vigfree,
        mock_file_manager,
    ):
        """Test successful execution of main pipeline."""
        # Setup mocks
        mock_fetch_odds.return_value = self.mock_raw_odds
        mock_process_target.return_value = self.mock_processed_odds
        mock_calc_vigfree.return_value = self.mock_vigfree_data
        mock_file_manager.return_value = Mock()

        # Run main
        main()

        # Verify pipeline executed
        mock_fetch_odds.assert_called_once()
        mock_process_target.assert_called_once_with(self.mock_raw_odds)
        mock_calc_vigfree.assert_called_once()
        
        # Verify all strategies were run (5 strategies)
        self.assertEqual(mock_run_strategy.call_count, 5)

    @patch("codebase.find_bets.find_bets.fetch_odds")
    def test_main_empty_raw_odds(self, mock_fetch_odds):
        """Test main when fetch_odds returns empty DataFrame."""
        mock_fetch_odds.return_value = pd.DataFrame()

        # Should complete without error
        main()

        mock_fetch_odds.assert_called_once()

    @patch("codebase.find_bets.find_bets.BetFileManager")
    @patch("codebase.find_bets.find_bets.calculate_vigfree_probabilities")
    @patch("codebase.find_bets.find_bets.process_target_odds_data")
    @patch("codebase.find_bets.find_bets.fetch_odds")
    def test_main_empty_processed_odds(
        self,
        mock_fetch_odds,
        mock_process_target,
        mock_calc_vigfree,
        mock_file_manager,
    ):
        """Test main when processed odds are empty."""
        mock_fetch_odds.return_value = self.mock_raw_odds
        mock_process_target.return_value = pd.DataFrame()

        # Should complete without error
        main()

        mock_fetch_odds.assert_called_once()
        mock_process_target.assert_called_once()
        # Should not proceed to vigfree calculation
        mock_calc_vigfree.assert_not_called()

    @patch("codebase.find_bets.find_bets.BetFileManager")
    @patch("codebase.find_bets.find_bets.calculate_vigfree_probabilities")
    @patch("codebase.find_bets.find_bets.process_target_odds_data")
    @patch("codebase.find_bets.find_bets.fetch_odds")
    def test_main_exception_handling(
        self,
        mock_fetch_odds,
        mock_process_target,
        mock_calc_vigfree,
        mock_file_manager,
    ):
        """Test that main raises exception on critical failure."""
        mock_fetch_odds.side_effect = Exception("API Error")

        with self.assertRaises(Exception):
            main()

    @patch("codebase.find_bets.find_bets.BetFileManager")
    @patch("codebase.find_bets.find_bets.calculate_vigfree_probabilities")
    @patch("codebase.find_bets.find_bets.process_target_odds_data")
    @patch("codebase.find_bets.find_bets.fetch_odds")
    @patch("codebase.find_bets.find_bets.run_betting_strategy")
    def test_main_all_strategies_executed(
        self,
        mock_run_strategy,
        mock_fetch_odds,
        mock_process_target,
        mock_calc_vigfree,
        mock_file_manager,
    ):
        """Test that all 5 betting strategies are executed."""
        # Setup mocks
        mock_fetch_odds.return_value = self.mock_raw_odds
        mock_process_target.return_value = self.mock_processed_odds
        mock_calc_vigfree.return_value = self.mock_vigfree_data
        mock_file_manager.return_value = Mock()

        # Run main
        main()

        # Verify all 5 strategies were called
        self.assertEqual(mock_run_strategy.call_count, 5)
        
        # Get all the strategy names that were called
        called_strategies = [call_args[0][0].name for call_args in mock_run_strategy.call_args_list]
        
        expected_strategies = [
            "Average Edge",
            "Z-Score Outliers",
            "Modified Z-Score Outliers",
            "Pinnacle Edge",
            "Random Bets",
        ]
        
        self.assertEqual(called_strategies, expected_strategies)

    @patch("codebase.find_bets.find_bets.BetFileManager")
    @patch("codebase.find_bets.find_bets.calculate_vigfree_probabilities")
    @patch("codebase.find_bets.find_bets.process_target_odds_data")
    @patch("codebase.find_bets.find_bets.fetch_odds")
    @patch("codebase.find_bets.find_bets.run_betting_strategy")
    def test_main_file_manager_initialized_once(
        self,
        mock_run_strategy,
        mock_fetch_odds,
        mock_process_target,
        mock_calc_vigfree,
        mock_file_manager_class,
    ):
        """Test that BetFileManager is initialized only once."""
        # Setup mocks
        mock_fetch_odds.return_value = self.mock_raw_odds
        mock_process_target.return_value = self.mock_processed_odds
        mock_calc_vigfree.return_value = self.mock_vigfree_data
        
        mock_file_manager_instance = Mock()
        mock_file_manager_class.return_value = mock_file_manager_instance

        # Run main
        main()

        # Verify BetFileManager was instantiated once
        mock_file_manager_class.assert_called_once()
        
        # Verify the same instance was passed to all strategy runs
        for call_args in mock_run_strategy.call_args_list:
            self.assertEqual(call_args[0][2], mock_file_manager_instance)

if __name__ == "__main__":
    unittest.main(verbosity=2)