# Paper Betting Tracker

This project tracks sports bets and results using Python scripts and CSV files. It fetches odds from The-Odds-API, analyzes profitable bets using several strategies, and logs results for further analysis.

## Betting Strategies
- Fair average odds: Calculates the vig-free (true) probability for an outcome from every bookmaker, then averages these probabilities to determine a consensus fair payout. Identifies betting opportunities where the best available odds offer higher payouts than this fair average suggests the outcome is worth.
- Z-score: Combines the fair average approach with statistical outlier detection. First filters for bets that exceed the fair average threshold, then applies an additional constraint requiring the best odds to be a certain distance away from the average to be considered profitable. This dual-filtering approach targets bets that are both fundamentally undervalued and anomalously priced.
- Modified Z-score: Uses the same dual-constraint approach as the Z-score strategy but employs a more robust statistical method. Instead of using mean and standard deviation (which can be skewed by extreme values), it uses median and median absolute deviation to identify outliers.
- Pinnacle edge: Compares available odds against Pinnacle Sportsbook's vig-free probabilities rather than a consensus average. Pinnacle is widely considered a "sharp" bookmaker with efficient pricing, so this strategy assumes Pinnacle's odds represent true market value and looks for opportunities where other bookmakers offer significantly better payouts.
- Random: Control strategy that randomly selects a small number of outcomes (0-5) and places bets on their best available odds, regardless of any mathematical analysis. This serves as a baseline to measure whether the analytical strategies actually outperform chance-based betting over time.

## Data Files Types
- *_bets.csv: A minimal file with only essential information for each bet placed.
- *_full.csv: A file with all bookmakers and odds that were available, not just the odds of the bookmaker placed.

## Project Structure
```
.
├── codebase/  # Main codebase containing all application logic
│   ├── analysis/                         # Package for analyzing results
│   │   ├── roi.py                        # Analyzes ROI for each data file
│   │   └── sample_size_confidence.py     # Creates ROI distributions that would be expected given win probabilities
│   ├── data/                 # Contains all bet and result CSV files. Files with "nc" indicate only bookmakers in NC are used for bet-finding.
│   │   ├── master_avg_bets.csv
│   │   ├── master_avg_full.csv
│   │   ├── master_mod_zscore_bets.csv
│   │   ├── master_mod_zscore_full.csv
│   │   ├── master_nc_avg_bets.csv
│   │   ├── master_nc_avg_full.csv
│   │   ├── master_nc_mod_zscore_bets.csv
│   │   ├── master_nc_mod_zscore_full.csv
│   │   ├── master_nc_pin_bets.csv
│   │   ├── master_nc_pin_full.csv
│   │   ├── master_nc_random_bets.csv
│   │   ├── master_nc_random_full.csv
│   │   ├── master_nc_zscore_bets.csv
│   │   ├── master_nc_zscore_full.csv
│   │   ├── master_pin_bets.csv
│   │   ├── master_pin_full.csv
│   │   ├── master_random_bets.csv
│   │   ├── master_random_full.csv
│   │   ├── master_zscore_bets.csv
│   │   └── master_zscore_full.csv
│   ├── fetch_odds/           # Package for fetching and organizing odds data
│   │   ├── __init__.py
│   │   ├── fetch_configs.py  # Configuration for odds fetching
│   │   └── fetch_odds.py     # Main odds fetching logic
│   ├── find_bets/            # Package for analyzing odds and finding profitable bets
│   │   ├── __init__.py
│   │   ├── betting_configs.py    # Betting analysis configuration
│   │   ├── betting_strategies.py # Core strategy analysis functions
│   │   ├── data_processing.py    # Data cleaning and validation
│   │   ├── file_management.py    # File operations and CSV handling
│   │   ├── find_bets.py          # Main orchestration and pipeline
│   │   └── summary_creation.py   # Summary generation functions
│   ├── results/              # Package for updating bet results
│   │   ├── __init__.py
│   │   ├── pinnacle_clv.py       # Pulls Pinnacle sportsbook's closing line odds for each bet that is placed before the start of the game
│   │   ├── results.py            # Main results updating logic
│   │   ├── results_configs.py    # Configuration for results fetching
│   │   ├── sportsdb_results.py   # Functions for pulling results from TheSportsDB
│   │   └── theodds_results.py    # Functions for pulling results from The-Odds-API
│   └── constants.py          # Shared constants across packages
│   └── __init__.py           
├── testing/                  # Test suite for all packages (UNDER CONSTRUCTION)
│   ├── fetch_odds/          
│   │   ├── test_fetch_odds_configs.py
│   │   └── test_fetch_odds.py
│   └── find_bets/            
│       ├── betting_strategies/
│       │   ├── test_betting_strategies.py
│       │   └── vf.csv
│       └── data_processing/
│       │   ├── pre_clean.csv
│       │   ├── pre_max_odds_check.csv
│       │   ├── pre_metadata.csv
│       │   ├── pre_min_bookmaker_check.csv
│       │   ├── pre_outcome_check.csv
│       │   ├── pre_prettify.csv
│       │   ├── processed.csv
│       │   ├── test_data_processing.py
│       │   └── unprocessed.csv
├── .github/                  # GitHub Actions workflows for automated running
│   └── workflows/
│       ├── hourly-bet-finder.yml
│       └── results.yml
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Fetch odds and analyze bets:
   ```bash
   python3 -m codebase.find_bets.find_bets
   ```
3. Update results:
   ```bash
   python3 -m codebase.results.results
   ```

4. View strategy profit/ROI:
   ```bash
   python3 -m codebase.analysis.roi
   ```

## Automation
- GitHub Actions workflows automatically runs bet finding script 3 times an hour and results updating script 1 time every 6 hours.
- Updated CSVs are committed to the repository.

## Author
- Andrew Smith
