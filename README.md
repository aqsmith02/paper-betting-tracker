# Paper Betting Tracker

This project tracks sports bets and results using Python scripts and CSV files. It fetches odds from The-Odds-API, analyzes profitable bets using several strategies, and logs results for further analysis.

## Features
- Fetches sports betting odds from The-Odds-API.
- Identifies profitable bets using the fair average odds, Z-score, modified Z-score, and Pinnacle edge strategies. Tracks a random betting strategy also for comparison.
- Organizes and saves bet and result data in CSV files under the `data/` folder.
- Automated workflows for updating bets and results.

## Project Structure
```
.
├── fetch_odds.py           # Fetches and organizes odds data
├── find_bets.py            # Analyzes odds and logs profitable bets
├── sportsdb_results.py     # Functions for pulling results from TheSportsDB
├── theodds_results.py      # Functions for pulling results from The-Odds-API
├── results.py              # Updates bet results
├── data/                   # Contains all bet and result CSV files
│   ├── master_avg_bets.csv
│   ├── master_avg_full.csv
│   ├── master_mod_zscore_bets.csv
│   ├── master_mod_zscore_full.csv
│   ├── master_pin_bets.csv
│   ├── master_pin_full.csv
│   ├── master_random_bets.csv
│   ├── master_random_full.csv
│   ├── master_zscore_bets.csv
│   ├── master_zscore_full.csv
├── .github/workflows/      # GitHub Actions workflows for automated running
│   ├── hourly-bet-finder.yml
│   └── results.yml
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
```

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Fetch odds and analyze bets:
   ```bash
   python fetch_odds.py
   python find_bets.py
   ```
3. Update results:
   ```bash
   python results.py
   ```

## Automation
- GitHub Actions workflows automatically runs bet finding script 4 times an hour and results updating script 1 time every 3 hours.
- Updated CSVs are committed to the repository.

## Author
- Andrew Smith