# Paper Betting Tracker

This project is an automated system for tracking and evaluating “paper” sports bets—bets that are recorded for analysis but do not involve real money. Using Python scripts and CSV files, the system fetches live odds from The-Odds-API, identifies potentially profitable bets using two strategies, and logs outcomes for downstream analysis.

Currently, the system considers only moneyline (head-to-head) bets and simulates placements exclusively on North Carolina sportsbooks, while evaluating all sports leagues supported by The-Odds-API. Data collection for the reported results began on October 11, 2025.

## Results

### Profit Over Time

The following showcase profit for each strategy since data collection has started. Bet sizing uses the 1/2 kelly criterion strategy. Bets are only placed when the EV is greater than 5%. The max bet size is 2.5 units. 

![Chart](analysis/output/profit_over_time/profit_over_time_all_strategies.png)
![Chart](analysis/output/profit_over_time/profit_over_time_average.png)
![Chart](analysis/output/profit_over_time/profit_over_time_average_with_modified_zscore_constraint.png)
![Chart](analysis/output/profit_over_time/profit_over_time_random_strategy.png)

### Hypothesis Testing

The following are Monte Carlo simulations run using the null hypothesis, that the EV of each bet is -5% (a typical number for a random bet). Using the equation **probability = (EV + 1) / odds**, we can calculate the probability of each bet placed in our dataset under the null hypothesis. We then randomly select whether each bet was a win or loss based on this probability, sum the profit, store the results, and repeat. After completion, we have a profit distribution under the null hypothesis.

![Chart](analysis/output/monte_carlo/monte_carlo_average.png)
![Chart](analysis/output/monte_carlo/monte_carlo_average_with_modified_zscore_constraint.png)

## Betting Strategies

### Fair Average Odds
Calculates the vig-free (true) probability for an outcome from every bookmaker, then averages these probabilities to determine a consensus fair payout. Identifies betting opportunities where the best available odds offer higher payouts than this fair average suggests the outcome is worth.

### Modified Z-Score
Combines the fair average approach with statistical outlier detection. First filters for bets that exceed the fair average threshold, then applies an additional constraint requiring the best odds to be a certain distance away from the average to be considered profitable. This dual-filtering approach targets bets that are both fundamentally undervalued and anomalously priced. The modified z-score is used over the traditional z-score because it is more sensitive to outliers in a small dataset.

### Random (Control)
Randomly selects a small number of outcomes (0–5) and places bets on their best available odds, regardless of any mathematical analysis. This serves as a baseline to measure whether analytical strategies outperform chance-based betting over time.


## Data Files Types

### *_bets.csv
A minimal file with only essential information for each bet placed.
### *_full.csv
A file with all bookmakers and odds that were available, not just the odds of the bookmaker placed.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Create an api_config file and add API keys:
   ```bash
   cp src/api_config.example.yaml src/api_config.yaml
   # Then edit src/api_config.yaml with your API keys and add it to .gitignore
   ```

3. Fetch odds and analyze bets:
   ```bash
   python3 -m src.find_bets.find_bets
   ```
   
4. Update results:
   ```bash
   python3 -m src.results.results
   ```
   
5. View strategy profit/ROI:
   ```bash
   python3 -m src.analysis.roi
   ```

## Automation

- GitHub Actions workflows automatically runs bet finding script 7 times an hour and results updating script 1 time every 6 hours.
- Updated CSVs are committed to the repository.

## Author

Andrew Smith
