"""
ev_analysis.py

Analyze the relationship between Expected Value and actual returns.
Validates whether EV predictions match reality.

Author: Andrew Smith
Date: February 2026
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EVAnalyzer:
    """Analyze Expected Value predictions vs actual results."""

    def __init__(self, df: pd.DataFrame, starting_bankroll: float = 100.0):
        """
        Initialize the EV analyzer.

        Args:
            df: DataFrame with betting data
            starting_bankroll: Initial bankroll for calculations
        """
        self.df = df.copy()
        self.starting_bankroll = starting_bankroll
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data with necessary calculations."""
        # Filter for completed bets only
        pending_statuses = ["Pending", "Not Found", "API Error"]
        self.df = self.df[~self.df["Result"].isin(pending_statuses)].copy()

        if len(self.df) == 0:
            print("Warning: No completed bets found")
            return

        # Determine if bet won
        self.df["Bet_Won"] = self.df.apply(
            lambda row: row["Result"] == row["Team"], axis=1
        )

        # Calculate Kelly and stake
        self.df["Fair_Probability"] = 1 / self.df["Fair Odds Average"]
        self.df["Implied_Probability"] = 1 / self.df["Best Odds"]

        # 1/2 Kelly sizing
        b = self.df["Best Odds"] - 1
        p = self.df["Fair_Probability"]
        q = 1 - p
        kelly = ((b * p - q) / b).clip(lower=0)
        self.df["Stake_Pct"] = (kelly / 2).clip(upper=0.05)

        # Calculate ROI per bet
        stake = self.df["Stake_Pct"] * self.starting_bankroll
        self.df["Profit_Loss"] = np.where(
            self.df["Bet_Won"], stake * (self.df["Best Odds"] - 1), -stake
        )
        self.df["ROI"] = (self.df["Profit_Loss"] / stake) * 100

        # Create EV bins
        self._create_ev_bins()

    def _create_ev_bins(self):
        """Create Expected Value bins for analysis."""
        # Define EV bins
        ev_bins = [
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.225,
            0.25,
            0.275,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            1.0,
        ]
        ev_labels = [
            "5-7.5%",
            "7.5-10%",
            "10-12.5%",
            "12.5-15%",
            "15-17.5%",
            "17.5-20%",
            "20-22.5%",
            "22.5-25%",
            "25-27.5%",
            "27.5-30%",
            "30-35%",
            "35-40%",
            "40-45%",
            "45-50%",
            "Above 50%",
        ]

        self.df["EV_Bin"] = pd.cut(
            self.df["Expected Value"],
            bins=ev_bins,
            labels=ev_labels,
            include_lowest=True,
        )

    def analyze_ev_vs_actual(self) -> pd.DataFrame:
        """
        Compare Expected Value bins to actual performance.

        Returns:
            DataFrame with EV bins and their actual performance
        """
        if len(self.df) == 0:
            return pd.DataFrame()

        analysis = (
            self.df.groupby("EV_Bin", observed=True)
            .agg(
                {
                    "Expected Value": ["mean", "min", "max"],
                    "ROI": ["mean", "std"],
                    "Bet_Won": ["sum", "count", "mean"],
                    "Best Odds": "mean",
                    "Profit_Loss": "sum",
                }
            )
            .round(4)
        )

        # Flatten column names
        analysis.columns = ["_".join(col).strip() for col in analysis.columns.values]
        analysis = analysis.rename(
            columns={
                "Expected Value_mean": "Avg_EV",
                "Expected Value_min": "Min_EV",
                "Expected Value_max": "Max_EV",
                "ROI_mean": "Actual_ROI",
                "ROI_std": "ROI_StdDev",
                "Bet_Won_sum": "Wins",
                "Bet_Won_count": "Total_Bets",
                "Bet_Won_mean": "Win_Rate",
                "Best Odds_mean": "Avg_Odds",
                "Profit_Loss_sum": "Total_Profit",
            }
        )

        # Convert to percentages
        analysis["Avg_EV"] = (analysis["Avg_EV"] * 100).round(2)
        analysis["Min_EV"] = (analysis["Min_EV"] * 100).round(2)
        analysis["Max_EV"] = (analysis["Max_EV"] * 100).round(2)
        analysis["Actual_ROI"] = analysis["Actual_ROI"].round(2)
        analysis["Win_Rate"] = (analysis["Win_Rate"] * 100).round(2)

        # Calculate EV accuracy (how close is actual ROI to predicted EV?)
        analysis["EV_Accuracy"] = (
            100 - abs(analysis["Actual_ROI"] - analysis["Avg_EV"])
        ).round(2)

        return analysis

    def calculate_calibration(self) -> Dict:
        """
        Calculate how well-calibrated the EV estimates are.

        Returns:
            Dictionary with calibration metrics
        """
        if len(self.df) == 0:
            return {}

        # Convert EV to ROI-equivalent percentage
        predicted_roi = self.df["Expected Value"] * 100
        actual_roi = self.df["ROI"]

        # Mean Absolute Error
        mae = abs(predicted_roi - actual_roi).mean()

        # Root Mean Square Error
        rmse = np.sqrt(((predicted_roi - actual_roi) ** 2).mean())

        # Correlation
        correlation = predicted_roi.corr(actual_roi)

        # Bias (are we systematically over/under-estimating?)
        bias = (predicted_roi - actual_roi).mean()

        return {
            "mean_absolute_error": round(mae, 2),
            "root_mean_square_error": round(rmse, 2),
            "correlation": round(correlation, 3),
            "bias": round(bias, 2),
            "interpretation": self._interpret_bias(bias),
        }

    def _interpret_bias(self, bias: float) -> str:
        """Interpret the bias value."""
        if abs(bias) < 2:
            return "Well-calibrated"
        elif bias > 0:
            return f"Overestimating by {abs(bias):.1f}% on average"
        else:
            return f"Underestimating by {abs(bias):.1f}% on average"

    def print_ev_analysis(self):
        """Print formatted EV analysis."""
        print("\n" + "=" * 80)
        print("EXPECTED VALUE ANALYSIS")
        print("=" * 80)

        # By bins
        print("\nPERFORMANCE BY EV BINS")
        print("-" * 80)
        ev_analysis = self.analyze_ev_vs_actual()
        if not ev_analysis.empty:
            print(ev_analysis.to_string())

        # Calibration
        print("\n\nCALIBRATION METRICS")
        print("-" * 80)
        calibration = self.calculate_calibration()
        if calibration:
            print(f"Mean Absolute Error: {calibration['mean_absolute_error']}%")
            print(f"Root Mean Square Error: {calibration['root_mean_square_error']}%")
            print(f"Correlation (EV vs ROI): {calibration['correlation']}")
            print(f"Bias: {calibration['bias']}%")
            print(f"Interpretation: {calibration['interpretation']}")

        print("\n" + "=" * 80 + "\n")

    def plot_ev_analysis(self, save_path: Optional[str] = None):
        """
        Create visualization of EV vs actual performance.

        Args:
            save_path: Optional path to save the plot
        """
        if len(self.df) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. EV Bins vs Actual ROI
        ax1 = axes[0, 0]
        ev_analysis = self.analyze_ev_vs_actual()
        if not ev_analysis.empty:
            x = range(len(ev_analysis))
            width = 0.35

            ax1.bar(
                [i - width / 2 for i in x],
                ev_analysis["Avg_EV"],
                width,
                label="Expected (EV)",
                alpha=0.8,
                color="steelblue",
            )
            ax1.bar(
                [i + width / 2 for i in x],
                ev_analysis["Actual_ROI"],
                width,
                label="Actual (ROI)",
                alpha=0.8,
                color="coral",
            )

            ax1.set_xlabel("EV Bin")
            ax1.set_ylabel("Return (%)")
            ax1.set_title("Expected vs Actual Returns by EV Bin")
            ax1.set_xticks(x)
            ax1.set_xticklabels(ev_analysis.index, rotation=45)
            ax1.legend()
            ax1.grid(axis="y", alpha=0.3)
            ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # 2. Scatter: Predicted vs Actual
        ax2 = axes[0, 1]
        predicted = self.df["Expected Value"] * 100
        actual = self.df["ROI"]

        ax2.scatter(predicted, actual, alpha=0.5, s=30)

        # Add diagonal line (perfect calibration)
        min_val = min(predicted.min(), actual.min())
        max_val = max(predicted.max(), actual.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Perfect Calibration",
            linewidth=2,
        )

        # Add trend line
        z = np.polyfit(predicted, actual, 1)
        p = np.poly1d(z)
        ax2.plot(
            predicted.sort_values(),
            p(predicted.sort_values()),
            "g-",
            label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
            linewidth=2,
        )

        ax2.set_xlabel("Expected Value (%)")
        ax2.set_ylabel("Actual ROI (%)")
        ax2.set_title("EV Calibration: Predicted vs Actual")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Win Rate by EV Bin
        ax3 = axes[1, 0]
        if not ev_analysis.empty:
            ax3.bar(
                range(len(ev_analysis)),
                ev_analysis["Win_Rate"],
                color="mediumseagreen",
                alpha=0.8,
            )
            ax3.set_xlabel("EV Bin")
            ax3.set_ylabel("Win Rate (%)")
            ax3.set_title("Win Rate by EV Bin")
            ax3.set_xticks(range(len(ev_analysis)))
            ax3.set_xticklabels(ev_analysis.index, rotation=45)
            ax3.grid(axis="y", alpha=0.3)
            ax3.axhline(
                y=50, color="red", linestyle="--", label="50% (Breakeven)", linewidth=1
            )
            ax3.legend()

        # 4. Sample Size by EV Bin
        ax4 = axes[1, 1]
        if not ev_analysis.empty:
            colors = [
                "lightcoral" if x < 30 else "lightgreen"
                for x in ev_analysis["Total_Bets"]
            ]
            ax4.bar(
                range(len(ev_analysis)),
                ev_analysis["Total_Bets"],
                color=colors,
                alpha=0.8,
            )
            ax4.set_xlabel("EV Bin")
            ax4.set_ylabel("Number of Bets")
            ax4.set_title("Sample Size by EV Bin")
            ax4.set_xticks(range(len(ev_analysis)))
            ax4.set_xticklabels(ev_analysis.index, rotation=45)
            ax4.grid(axis="y", alpha=0.3)
            ax4.axhline(
                y=30,
                color="orange",
                linestyle="--",
                label="Minimum for significance",
                linewidth=1,
            )
            ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


def analyze_ev_performance(
    csv_path: str,
    starting_bankroll: float = 10000.0,
    plot: bool = True,
    save_plot: Optional[str] = None,
):
    """
    Analyze EV performance from a CSV file.

    Args:
        csv_path: Path to the CSV file
        starting_bankroll: Starting bankroll
        plot: Whether to create plots
        save_plot: Optional path to save plot

    Returns:
        EVAnalyzer instance
    """
    df = pd.read_csv(csv_path)
    analyzer = EVAnalyzer(df, starting_bankroll)

    analyzer.print_ev_analysis()

    if plot:
        analyzer.plot_ev_analysis(save_plot)

    return analyzer


if __name__ == "__main__":
    csv_paths = [
        "data/nc_avg_minimal.csv",
        "data/nc_mod_zscore_minimal.csv",
        "data/nc_random_minimal.csv",
    ]

    for csv_path in csv_paths:
        analyzer = analyze_ev_performance(csv_path, 100)
