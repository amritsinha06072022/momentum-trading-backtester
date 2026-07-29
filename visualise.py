"""
visualise.py — Phase 5: Visualisation and Reporting
====================================================

Turns the numerical results of Phases 1-4 into the two charts a recruiter or
reviewer actually looks at. Each chart is both shown on screen (interactive
window in PyCharm) and saved to a PNG for embedding in the project README.

Charts produced
---------------
1. Price & Signals  (price_signals.png)
   The asset's closing price overlaid with the 50- and 200-day moving averages,
   with green up-arrows where the strategy actually BOUGHT and red down-arrows
   where it SOLD. Markers use the engine's EXECUTED trades (shifted +1 day for
   the lookahead fix, at real fill prices), so they show what genuinely happened,
   not the raw same-day signals.

2. Equity Curve  (equity_curve.png)
   The strategy's portfolio value over time plotted against the buy-and-hold
   benchmark, starting from the same initial capital. This is the single chart
   that tells the project's story: whether the strategy's activity beat simply
   owning the asset. Drawdown periods (equity below its running peak) are shaded
   so the worst-case pain is visible at a glance.

Note: the project .gitignore excludes *.png. To embed these two charts in your
README, force-add them:  git add -f price_signals.png equity_curve.png

Dependencies: pandas, numpy, matplotlib
"""

import matplotlib.pyplot as plt
import pandas as pd

# A clean, readable style. 'seaborn-v0_8-darkgrid' ships with matplotlib;
# fall back gracefully if the name differs across versions.
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")


def plot_price_signals(df: pd.DataFrame,
                       trades: pd.DataFrame,
                       ticker: str = "AAPL",
                       save_path: str = "price_signals.png") -> None:
    """Plot price + moving averages with executed buy/sell markers.

    df     : DataFrame with 'Close', 'SMA_fast', 'SMA_slow' (from Phase 2).
    trades : executed-trade log from the engine (index=date, 'Type', 'Price').
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df["Close"], label="Close Price",
            color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.plot(df.index, df["SMA_fast"], label="50-day SMA",
            color="#ff7f0e", linewidth=1.3)
    ax.plot(df.index, df["SMA_slow"], label="200-day SMA",
            color="#9467bd", linewidth=1.3)

    # Executed trades: green ^ for buys, red v for sells, placed at fill price.
    if not trades.empty:
        buys = trades[trades["Type"] == "BUY"]
        sells = trades[trades["Type"] == "SELL"]
        ax.scatter(buys.index, buys["Price"], marker="^", s=160,
                   color="#2ca02c", edgecolor="black", linewidth=0.6,
                   zorder=5, label="Buy")
        ax.scatter(sells.index, sells["Price"], marker="v", s=160,
                   color="#d62728", edgecolor="black", linewidth=0.6,
                   zorder=5, label="Sell")

    ax.set_title(f"{ticker}: Price, Moving Averages & Trade Signals",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def plot_equity_curve(results: pd.DataFrame,
                      bh_equity: pd.Series,
                      initial_capital: float = 10_000.0,
                      save_path: str = "equity_curve.png") -> None:
    """Plot strategy equity vs. buy-and-hold, with drawdown periods shaded.

    results   : per-day results from the engine (needs 'Equity' column).
    bh_equity : buy-and-hold equity curve (from metrics.buy_and_hold).
    """
    strat = results["Equity"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(strat.index, strat, label="Strategy",
            color="#2ca02c", linewidth=1.6)
    ax.plot(bh_equity.index, bh_equity, label="Buy & Hold",
            color="#1f77b4", linewidth=1.6, alpha=0.8)

    # Starting capital reference line.
    ax.axhline(initial_capital, color="grey", linestyle="--",
               linewidth=0.9, alpha=0.7, label=f"Start (${initial_capital:,.0f})")

    # Shade the strategy's drawdown periods (equity below its running peak).
    running_max = strat.cummax()
    ax.fill_between(strat.index, strat, running_max,
                    where=(strat < running_max), color="#d62728",
                    alpha=0.15, label="Strategy drawdown")

    ax.set_title("Equity Curve: Strategy vs. Buy & Hold",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    from data_pipeline import fetch_data, clean_data
    from signals import generate_signals
    from engine import run_backtest
    from metrics import buy_and_hold

    TICKER = "AAPL"

    data = generate_signals(clean_data(fetch_data(TICKER, period="10y")))
    results, trades = run_backtest(data)
    bh_equity = buy_and_hold(data, initial_capital=10_000.0)

    plot_price_signals(data, trades, ticker=TICKER)
    plot_equity_curve(results, bh_equity, initial_capital=10_000.0)

    # Open both interactive windows (PyCharm shows these on-screen).
    plt.show()