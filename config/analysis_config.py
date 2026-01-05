"""
Configuration for betting analysis.

This module contains all configuration parameters used across analysis modules.
Edit values here to change behavior across all analyses.
"""

# ============================================================================
# Kelly Criterion Parameters
# ============================================================================

# Fraction of Kelly to bet (0.5 = half Kelly, reduces variance)
KELLY_FRACTION = 0.5

# Only bet on opportunities with EV in this range
MIN_EV_THRESHOLD = 0.05  # 5% minimum expected value
MAX_EV_THRESHOLD = 0.35  # 35% maximum expected value (filter out extremes)

# Maximum bet size as percentage of bankroll
MAX_BET_MULTIPLIER = 2.5  # 2.5% max bet size

# Z-score threshold to trigger max bet (high confidence bets)
ZSCORE_MAX_BET_THRESHOLD = 3.5


# ============================================================================
# Monte Carlo Simulation Parameters
# ============================================================================

# Expected value under null hypothesis (bookmaker edge)
NULL_EV = -0.05  # -5% EV (typical vig)

# Number of simulations to run
N_SIMULATIONS = 10000

# Random seed for reproducibility (None = random seed each run)
RANDOM_SEED = 1

# Significance levels for hypothesis testing
ALPHA_LEVELS = [0.05, 0.01]  # 5% and 1% significance


# ============================================================================
# Bin Analysis Parameters
# ============================================================================

# Expected Value bins (as decimals, e.g., 0.05 = 5%)
EV_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 1.0]
EV_LABELS = ['0-2%', '2-5%', '5-10%', '10-15%', '15-20%', '20%+']

# Z-Score bins
ZSCORE_BINS = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 10.0]
ZSCORE_LABELS = ['0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5+']


# ============================================================================
# Visualization Parameters
# ============================================================================

# Figure size for plots (width, height in inches)
FIGURE_SIZE_DEFAULT = (14, 8)
FIGURE_SIZE_COMPARISON = (14, 8)
FIGURE_SIZE_MONTE_CARLO = (14, 5)

# DPI for saved figures
DPI = 300

# Color scheme for strategy plots
COLORS = [
    '#2E86AB',  # Blue
    '#A23B72',  # Purple
    '#F18F01',  # Orange
    '#C73E1D',  # Red
    '#6A994E',  # Green
]

# Date format for x-axis labels
DATE_FORMAT = '%Y-%m-%d'

# Font sizes
FONT_SIZE_TITLE = 16
FONT_SIZE_AXIS_LABEL = 13
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOTATION = 10
FONT_SIZE_DATE_STAMP = 14


# ============================================================================
# Output Paths
# ============================================================================

# Base directory for all output files
OUTPUT_BASE_DIR = 'analysis/output'

# Subdirectories for different analysis types
OUTPUT_PROFIT_CHARTS = 'profit_over_time'
OUTPUT_MONTE_CARLO = 'monte_carlo'
OUTPUT_BIN_ANALYSIS = 'bin_analysis'

# File naming patterns
FILE_PATTERN_PROFIT_CHART = 'profit_over_time_{strategy_name}.png'
FILE_PATTERN_MONTE_CARLO_CHART = 'monte_carlo_{strategy_name}.png'
FILE_PATTERN_MONTE_CARLO_RESULTS = 'monte_carlo_results.csv'
FILE_PATTERN_COMPARISON_CHART = 'profit_over_time_all_strategies.png'


# ============================================================================
# Data Parameters
# ============================================================================

# Date/time columns in data files
DATETIME_COLUMNS = ['Start Time', 'Scrape Time']

# Result columns
RESULT_COLUMN = 'Result'
TEAM_COLUMN = 'Team'

# Date format in data files (for parsing, None = infer automatically)
DATETIME_FORMAT = None


# ============================================================================
# Utility Functions
# ============================================================================

from pathlib import Path


def get_output_path(analysis_type, filename=None):
    """
    Get output path for a specific analysis type.
    
    Args:
        analysis_type: Type of analysis ('profit_charts', 'monte_carlo', 'bin_analysis')
        filename: Optional filename to append to path
        
    Returns:
        Path object for output location
        
    Examples:
        >>> get_output_path('profit_charts')
        Path('analysis/output/profit_over_time')
        >>> get_output_path('monte_carlo', 'results.csv')
        Path('analysis/output/monte_carlo/results.csv')
    """
    base = Path(OUTPUT_BASE_DIR)
    
    type_dirs = {
        'profit_charts': OUTPUT_PROFIT_CHARTS,
        'monte_carlo': OUTPUT_MONTE_CARLO,
        'bin_analysis': OUTPUT_BIN_ANALYSIS,
    }
    
    if analysis_type not in type_dirs:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    path = base / type_dirs[analysis_type]
    
    # Create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    
    if filename:
        path = path / filename
        
    return path


def format_output_filename(pattern_name, **kwargs):
    """
    Format an output filename using configured patterns.
    
    Args:
        pattern_name: Name of pattern ('profit_chart', 'monte_carlo_chart', etc.)
        **kwargs: Values to substitute in pattern (e.g., strategy_name='Average')
        
    Returns:
        Formatted filename string
        
    Examples:
        >>> format_output_filename('profit_chart', strategy_name='Average')
        'profit_over_time_average.png'
    """
    patterns = {
        'profit_chart': FILE_PATTERN_PROFIT_CHART,
        'monte_carlo_chart': FILE_PATTERN_MONTE_CARLO_CHART,
        'monte_carlo_results': FILE_PATTERN_MONTE_CARLO_RESULTS,
        'comparison_chart': FILE_PATTERN_COMPARISON_CHART,
    }
    
    if pattern_name not in patterns:
        raise ValueError(f"Unknown file pattern: {pattern_name}")
    
    pattern = patterns[pattern_name]
    
    # Format strategy names (lowercase, replace spaces with underscores)
    if 'strategy_name' in kwargs:
        kwargs['strategy_name'] = kwargs['strategy_name'].lower().replace(' ', '_')
    
    return pattern.format(**kwargs)