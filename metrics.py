"""
metrics.py — Phase 4: Quantitative Performance Metrics
=======================================================

Proves — or disproves — whether the strategy was actually any good. Total profit
is a vanity number; quants evaluate RISK-ADJUSTED performance. This module
computes four metrics for both the strategy and a buy-and-hold benchmark, so
every result is judged against the honest "do nothing" alternative.

Metrics
-------
Cumulative Return : total % gained/lost over the period.
Sharpe Ratio      : annualised excess return per unit of volatility. Higher is
                    better; > 1 is generally considered good. Rewards smooth
                    growth and penalises a bumpy ride.
Maximum Drawdown  : worst peak-to-trough decline (%). Measures worst-case pain —
                    the metric that decides whether you could stomach holding.
Win Rate          : % of closed round-trip trades that were profitable. Note a
                    LOW win rate is normal and healthy for trend-following, where
                    a few large winners outweigh many small losses.

Benchmark
---------
The buy-and-hold curve invests all capital at the first close and holds to the
end. Comparing against it answers the only question that matters: did the
strategy's activity beat simply owning the asset? Often it does not on raw
return, but wins on drawdown — trading return for a smoother, safer ride.

Public functions
----------------
cumulative_return(equity)      -> float (%)
sharpe_ratio(returns, rf, ...) -> float
max_drawdown(equity)           -> float (%)
analyze_trades(trades, close)  -> (round_trips_df, stats_dict)
buy_and_hold(df, capital)      -> pd.Series (equity curve)
performance_report(results, trades, df, capital, rf) -> None (prints comparison)

Dependencies: pandas, numpy
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # annualisation factor for daily data


def cumulative_return(equity: pd.Series) -> float:
    """Total percentage change from first to last value of an equity curve."""
    return (equity.iloc[-1] / equity.iloc[0] - 1) * 100


def sharpe_ratio(returns: pd.Series,
                 rf_annual: float = 0.0,
                 periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio from a series of periodic (daily) returns.

    rf_annual : annual risk-free rate (e.g. 0.04 for 4%). Defaults to 0, which
    is transparent and makes the strategy-vs-benchmark comparison clean; set it
    to the prevailing T-bill rate for a more textbook-standard figure.
    """
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    rf_daily = rf_annual / periods
    excess = returns - rf_daily
    return (excess.mean() / returns.std()) * np.sqrt(periods)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline of an equity curve, as a negative %."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min() * 100


def analyze_trades(trades: pd.DataFrame, last_close: float):
    """Pair BUY->SELL round trips and compute P&L, win rate, and any open position.

    Returns (round_trips_df, stats_dict). The final position may be still-open
    (an unmatched BUY); it is marked to market at last_close, reported as
    unrealized, and EXCLUDED from the win-rate denominator (it isn't closed yet).
    """
    if trades.empty:
        return pd.DataFrame(), {"closed": 0, "wins": 0,
                                "win_rate": float("nan"), "open_unrealized": 0.0}

    t = trades.reset_index()
    round_trips = []
    i = 0
    while i < len(t):
        row = t.iloc[i]
        if row["Type"] == "BUY":
            buy = row
            if i + 1 < len(t) and t.iloc[i + 1]["Type"] == "SELL":
                sell = t.iloc[i + 1]
                pnl = (sell["Price"] - buy["Price"]) * buy["Shares"] \
                      - buy["Cost"] - sell["Cost"]
                round_trips.append({
                    "Entry": buy["Date"], "Exit": sell["Date"],
                    "BuyPx": buy["Price"], "SellPx": sell["Price"],
                    "PnL": pnl, "Win": pnl > 0, "Status": "closed",
                })
                i += 2
            else:
                pnl = (last_close - buy["Price"]) * buy["Shares"] - buy["Cost"]
                round_trips.append({
                    "Entry": buy["Date"], "Exit": None,
                    "BuyPx": buy["Price"], "SellPx": last_close,
                    "PnL": pnl, "Win": pnl > 0, "Status": "OPEN",
                })
                i += 1
        else:
            i += 1  # a SELL with no preceding BUY should never happen; skip safely

    rt = pd.DataFrame(round_trips)
    closed = rt[rt["Status"] == "closed"]
    n = len(closed)
    wins = int(closed["Win"].sum())
    open_pnl = float(rt[rt["Status"] == "OPEN"]["PnL"].sum())

    return rt, {
        "closed": n,
        "wins": wins,
        "win_rate": (wins / n * 100) if n else float("nan"),
        "open_unrealized": open_pnl,
    }


def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.Series:
    """Equity curve for investing all capital at the first close and holding."""
    close = df["Close"]
    return initial_capital * (close / close.iloc[0])


def performance_report(results: pd.DataFrame,
                       trades: pd.DataFrame,
                       df: pd.DataFrame,
                       initial_capital: float = 10_000.0,
                       rf_annual: float = 0.0) -> None:
    """Print a side-by-side strategy-vs-buy-and-hold performance comparison."""
    # Strategy metrics
    strat_equity = results["Equity"]
    strat_returns = strat_equity.pct_change()
    strat_cum = cumulative_return(strat_equity)
    strat_sharpe = sharpe_ratio(strat_returns, rf_annual)
    strat_dd = max_drawdown(strat_equity)

    # Benchmark metrics
    bh_equity = buy_and_hold(df, initial_capital)
    bh_returns = bh_equity.pct_change()
    bh_cum = cumulative_return(bh_equity)
    bh_sharpe = sharpe_ratio(bh_returns, rf_annual)
    bh_dd = max_drawdown(bh_equity)

    # Trade stats
    last_close = df["Close"].iloc[-1]
    rt, stats = analyze_trades(trades, last_close)

    print("=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"{'Metric':<22}{'Strategy':>16}{'Buy & Hold':>16}")
    print("-" * 60)
    print(f"{'Cumulative Return':<22}{strat_cum:>15.2f}%{bh_cum:>15.2f}%")
    print(f"{'Sharpe Ratio':<22}{strat_sharpe:>16.2f}{bh_sharpe:>16.2f}")
    print(f"{'Max Drawdown':<22}{strat_dd:>15.2f}%{bh_dd:>15.2f}%")
    print(f"{'Final Equity':<22}${strat_equity.iloc[-1]:>14,.0f}"
          f"${bh_equity.iloc[-1]:>14,.0f}")
    print("-" * 60)
    print(f"Win rate:        {stats['win_rate']:.1f}% "
          f"({stats['wins']}/{stats['closed']} closed trades)")
    print(f"Open position (unrealized): ${stats['open_unrealized']:,.2f}")
    print("=" * 60)

    print("\n--- Round-trip trade log ---")
    if not rt.empty:
        show = rt.copy()
        show["PnL"] = show["PnL"].map(lambda x: f"${x:,.2f}")
        print(show.to_string(index=False))


if __name__ == "__main__":
    from data_pipeline import fetch_data, clean_data
    from signals import generate_signals
    from engine import run_backtest

    data = generate_signals(clean_data(fetch_data("AAPL", period="10y")))
    results, trades = run_backtest(data)

    performance_report(results, trades, data, initial_capital=10_000.0)

