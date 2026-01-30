"""
test_file_management.py

Pytest test suite for file_management.py module.

Author: Andrew Smith
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
from src.find_bets.file_management import (
    _start_date_from_timestamp,
    _filter_best_bets_only,
    _remove_duplicates,
    _align_column_schemas,
    save_betting_data,
)


class TestStartDateFromTimestamp:
    """Test _start_date_from_timestamp function."""
    
    def test_converts_string_timestamp(self):
        """Should convert string timestamp to YYYY-MM-DD format."""
        timestamp = '2024-01-15 19:30:00'
        result = _start_date_from_timestamp(timestamp)
        assert result == '2024-01-15'

    def test_converts_edge_string_timestamp(self):
        """Should convert string timestamp to YYYY-MM-DD format."""
        timestamp = '2024-01-15 00:00:00'
        result = _start_date_from_timestamp(timestamp)
        assert result == '2024-01-15'
    
    def test_converts_datetime_object(self):
        """Should convert datetime object to YYYY-MM-DD format."""
        timestamp = datetime(2024, 1, 15, 19, 30, 0)
        result = _start_date_from_timestamp(timestamp)
        assert result == '2024-01-15'
    
    def test_strips_time_component(self):
        """Should strip time component, keeping only date."""
        timestamp = '2024-01-15 23:59:59'
        result = _start_date_from_timestamp(timestamp)
        assert result == '2024-01-15'



class TestFilterBestBetsOnly:
    """Test _filter_best_bets_only function."""
    
    def test_keeps_highest_score_per_match(self):
        """Should keep only the bet with highest score for each match."""
        df = pd.DataFrame({
            'Match': ['Game 1', 'Game 1', 'Game 2'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00'] * 3),
            'Expected Value': [0.10, 0.20, 0.15],
        })
        
        result = _filter_best_bets_only(df, 'Expected Value')
        
        # Game 1 appears twice, should keep only the 0.20 EV bet
        game1_bets = result[result['Match'] == 'Game 1']
        assert len(game1_bets) == 1
        assert game1_bets['Expected Value'].iloc[0] == 0.20
    
    def test_uses_match_and_start_time_as_key(self):
        """Should use both Match and Start Time to identify duplicates."""
        df = pd.DataFrame({
            'Match': ['Game 1', 'Game 1', 'Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00', '2024-01-15 19:00', '2024-01-16 19:00']),
            'Expected Value': [0.10, 0.15, 0.20],
        })
        
        result = _filter_best_bets_only(df, 'Expected Value')
        
        # Should keep 2 bets: best from Jan 15 and the one from Jan 16
        assert len(result) == 2
        assert 0.15 in result['Expected Value'].values  # Best from Jan 15
        assert 0.20 in result['Expected Value'].values  # Only one from Jan 16
    
    def test_handles_empty_dataframe(self):
        """Should return empty DataFrame when input is empty."""
        df = pd.DataFrame()
        result = _filter_best_bets_only(df, 'Expected Value')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_works_with_different_score_columns(self):
        """Should work with any specified score column."""
        df = pd.DataFrame({
            'Match': ['Game 1', 'Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00'] * 2),
            'Expected Value': [0.20, 0.10],
            'Modified Z-Score': [2.0, 3.5],
        })
        
        # Using Expected Value
        result_ev = _filter_best_bets_only(df, 'Expected Value')
        assert result_ev['Expected Value'].iloc[0] == 0.20
        
        # Using Modified Z-Score
        result_z = _filter_best_bets_only(df, 'Modified Z-Score')
        assert result_z['Modified Z-Score'].iloc[0] == 3.5


class TestRemoveDuplicates:
    """Test _remove_duplicates function."""
    
    def test_removes_exact_duplicates(self):
        """Should remove bets that already exist (same Match and Start Date)."""
        existing = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00']),
            'Team': ['Team A'],
        })
        
        new = pd.DataFrame({
            'Match': ['Game 1', 'Game 2'],
            'Start Time': pd.to_datetime(['2024-01-15 19:30', '2024-01-16 20:00']),
            'Team': ['Team A', 'Team B'],
        })
        
        result = _remove_duplicates(existing, new)
        
        # Game 1 on same date should be removed, Game 2 should remain
        assert len(result) == 1
        assert result['Match'].iloc[0] == 'Game 2'
    
    def test_keeps_same_match_different_dates(self):
        """Should keep bets for same match on different dates."""
        existing = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00']),
        })
        
        new = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-16 19:00']),
        })
        
        result = _remove_duplicates(existing, new)
        
        # Different dates, should keep
        assert len(result) == 1
    
    def test_returns_all_new_when_existing_empty(self):
        """Should return all new data when existing is empty."""
        existing = pd.DataFrame()
        
        new = pd.DataFrame({
            'Match': ['Game 1', 'Game 2'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00', '2024-01-16 20:00']),
        })
        
        result = _remove_duplicates(existing, new)
        
        assert len(result) == 2


class TestAlignColumnSchemas:
    """Test _align_column_schemas function."""
    
    def test_inserts_new_columns_before_marker(self):
        """Should insert new columns before the INSERT_BEFORE_COLUMN."""
        existing = pd.DataFrame({
            'ID': [1],
            'Match': ['Game 1'],
            'Best Bookmaker': ['Bookmaker A'],
            'Result': [1],
        })
        
        new = pd.DataFrame({
            'ID': [2],
            'Match': ['Game 2'],
            'New Column': [100],
            'Best Bookmaker': ['Bookmaker A'],
            'Result': [0],
        })
        
        result = _align_column_schemas(existing, new)
        
        assert result == ['ID', 'Match', 'New Column', 'Best Bookmaker', 'Result']
    
    def test_returns_new_columns_when_existing_empty(self):
        """Should return new columns when existing DataFrame is empty."""
        existing = pd.DataFrame()
        new = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        
        result = _align_column_schemas(existing, new)
        
        assert result == ['A', 'B', 'C']
    
    def test_returns_existing_columns_when_new_empty(self):
        """Should return existing columns when new DataFrame is empty."""
        existing = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        new = pd.DataFrame()
        
        result = _align_column_schemas(existing, new)
        
        assert result == ['A', 'B', 'C']
    
    def test_handles_no_new_columns(self):
        """Should handle case where new df has no new columns."""
        existing = pd.DataFrame({'A': [1], 'B': [2], 'Best Bookmaker': ['Bookmaker A'], 'Result': [3]})
        new = pd.DataFrame({'A': [4], 'B': [5], 'Best Bookmaker': ['Bookmaker A'], 'Result': [6]})
        
        result = _align_column_schemas(existing, new)
        
        # Should just return existing columns
        assert result == ['A', 'B', 'Best Bookmaker', 'Result']


class TestSaveBettingData:
    """Test save_betting_data function."""
    
    def test_saves_to_correct_filename(self):
        """Should save to the specified filename."""
        existing = pd.DataFrame()
        new = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00']),
            'Expected Value': [0.15],
            'Result': [np.nan],
        })
        
        with patch('src.find_bets.file_management.pd.DataFrame.to_csv') as mock_to_csv:
            save_betting_data(existing, new, 'my_bets.csv', 'Expected Value')
            
            # Check filename in call args
            assert mock_to_csv.call_args[0][0] == 'my_bets.csv'
    
    def test_saves_without_index(self):
        """Should save with index=False."""
        existing = pd.DataFrame()
        new = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00']),
            'Expected Value': [0.15],
            'Result': [np.nan],
        })
        
        with patch('src.find_bets.file_management.pd.DataFrame.to_csv') as mock_to_csv:
            save_betting_data(existing, new, 'test.csv', 'Expected Value')
            
            # Check index=False in call args
            assert mock_to_csv.call_args[1]['index'] == False
    
    def test_handles_empty_new_data(self):
        """Should do nothing when new data is empty."""
        existing = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00']),
        })
        
        new = pd.DataFrame()
        
        with patch('src.find_bets.file_management.pd.DataFrame.to_csv') as mock_to_csv:
            save_betting_data(existing, new, 'test.csv', 'Expected Value')
            
            # Should not call to_csv when new data is empty
            mock_to_csv.assert_not_called()
    
    def test_prints_when_print_bets_true(self, capsys):
        """Should print bet information when print_bets=True."""
        existing = pd.DataFrame()
        new = pd.DataFrame({
            'Match': ['Game 1', 'Game 2'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00', '2024-01-16 20:00']),
            'Team': ['Team A', 'Team B'],
            'Expected Value': [0.15, 0.20],
            'Result': [np.nan, np.nan],
        })
        
        with patch('src.find_bets.file_management.pd.DataFrame.to_csv'):
            save_betting_data(existing, new, 'test.csv', 'Expected Value', print_bets=True)
            
            captured = capsys.readouterr()
            assert 'bets found' in captured.out
    
    def test_does_not_print_when_print_bets_false(self, capsys):
        """Should not print when print_bets=False."""
        existing = pd.DataFrame()
        new = pd.DataFrame({
            'Match': ['Game 1'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00']),
            'Team': ['Team A'],
            'Expected Value': [0.15],
            'Result': [np.nan],
        })
        
        with patch('src.find_bets.file_management.pd.DataFrame.to_csv'):
            save_betting_data(existing, new, 'test.csv', 'Expected Value', print_bets=False)
            
            captured = capsys.readouterr()
            # Should not have printed anything about bets
            assert 'bets found' not in captured.out