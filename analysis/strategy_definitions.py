"""
Strategy definitions and configurations.

This module contains all betting strategy configurations used across analysis files.
This is the single source of truth for strategy definitions - all other modules
should import from here to ensure consistency.

Strategy Types:
- Average: Uses average of multiple bookmaker odds as fair odds
- Modified Zscore: Average with modified z-score filtering
- Pinnacle: Uses Pinnacle closing odds as fair odds benchmark
- Zscore: Average with standard z-score filtering  
- Random: Control/baseline strategy with random selections
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BettingStrategy:
    """
    Configuration for a betting strategy.
    
    Attributes:
        name: Human-readable strategy name for display
        path: Path to CSV file containing betting data
        odds_column: Column name for best available odds
        fair_odds_column: Column name for fair odds estimate (None for random)
        ev_column: Column name for expected value (None for random)
        zscore_column: Column name for z-score filtering (None if not used)
    """
    name: str
    path: str
    odds_column: str
    fair_odds_column: Optional[str] = None
    ev_column: Optional[str] = None
    zscore_column: Optional[str] = None
    
    @property
    def has_ev(self) -> bool:
        """Check if strategy has expected value calculations."""
        return self.ev_column is not None
    
    @property
    def has_zscore(self) -> bool:
        """Check if strategy uses z-score filtering."""
        return self.zscore_column is not None
    
    @property
    def is_random(self) -> bool:
        """Check if this is a random/control strategy."""
        return self.ev_column is None and self.fair_odds_column is None
    
    @property
    def uses_kelly(self) -> bool:
        """Check if strategy can use Kelly criterion (needs fair odds)."""
        return self.fair_odds_column is not None


# ============================================================================
# ALL STRATEGIES
# ============================================================================

ALL_STRATEGIES = [
    BettingStrategy(
        name="Average",
        path="data/master_nc_avg_full.csv",
        odds_column="Best Odds",
        fair_odds_column="Fair Odds Avg",
        ev_column="Expected Value",
    ),
    BettingStrategy(
        name="Average With Modified Zscore Constraint",
        path="data/master_nc_mod_zscore_full.csv",
        odds_column="Best Odds",
        fair_odds_column="Fair Odds Avg",
        ev_column="Expected Value",
        zscore_column="Modified Z Score",
    ),
    BettingStrategy(
        name="Random Strategy",
        path="data/master_nc_random_full.csv",
        odds_column="Best Odds",
    ),
]


# ============================================================================
# STRATEGY SUBSETS FOR SPECIFIC ANALYSES
# ============================================================================

def get_strategies_with_ev() -> List[BettingStrategy]:
    """
    Get strategies that have expected value calculations.
    
    Used for: Kelly criterion analysis, EV bin analysis, Monte Carlo with EV filtering
    """
    return [s for s in ALL_STRATEGIES if s.has_ev]


def get_strategies_with_zscore() -> List[BettingStrategy]:
    """
    Get strategies that use z-score filtering.
    
    Used for: Z-score bin analysis, comparing z-score vs non-zscore strategies
    """
    return [s for s in ALL_STRATEGIES if s.has_zscore]


def get_strategies_for_kelly() -> List[BettingStrategy]:
    """
    Get strategies that can use Kelly criterion betting.
    
    Used for: ROI comparison (flat vs Kelly), profit over time with Kelly
    """
    return [s for s in ALL_STRATEGIES if s.uses_kelly]


def get_random_strategy() -> BettingStrategy:
    """
    Get the random/control strategy for baseline comparisons.
    
    Returns:
        The random strategy configuration
    """
    random_strats = [s for s in ALL_STRATEGIES if s.is_random]
    if not random_strats:
        raise ValueError("No random strategy found in ALL_STRATEGIES")
    return random_strats[0]


def get_non_random_strategies() -> List[BettingStrategy]:
    """
    Get all strategies except the random control.
    
    Used for: Analyses where random strategy should be excluded or treated separately
    """
    return [s for s in ALL_STRATEGIES if not s.is_random]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_strategy_by_name(name: str) -> Optional[BettingStrategy]:
    """
    Get a strategy by its name.
    
    Args:
        name: Strategy name (case-insensitive partial match)
    
    Returns:
        Matching strategy or None if not found
    """
    name_lower = name.lower()
    for strategy in ALL_STRATEGIES:
        if name_lower in strategy.name.lower():
            return strategy
    return None


def print_strategy_summary():
    """Print a summary of all available strategies."""
    print("=" * 80)
    print("AVAILABLE BETTING STRATEGIES")
    print("=" * 80)
    print(f"{'Name':<45} {'EV':<5} {'Z-Score':<10} {'Kelly':<8}")
    print("-" * 80)
    
    for strategy in ALL_STRATEGIES:
        ev_marker = "✓" if strategy.has_ev else "✗"
        zscore_marker = "✓" if strategy.has_zscore else "✗"
        kelly_marker = "✓" if strategy.uses_kelly else "✗"
        print(f"{strategy.name:<45} {ev_marker:<5} {zscore_marker:<10} {kelly_marker:<8}")
    
    print("=" * 80)
    print(f"Total Strategies: {len(ALL_STRATEGIES)}")
    print(f"With EV: {len(get_strategies_with_ev())}")
    print(f"With Z-Score: {len(get_strategies_with_zscore())}")
    print(f"Kelly Compatible: {len(get_strategies_for_kelly())}")
    print("=" * 80)


if __name__ == "__main__":
    # Print summary when run directly
    print_strategy_summary()