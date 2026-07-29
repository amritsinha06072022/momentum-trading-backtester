"""
engine.py — Phase 3: The Backtesting Engine
============================================

Simulates the passage of time day-by-day, executing the Buy/Sell signals from
Phase 2 against historical prices while tracking cash and holdings. Produces an
equity curve (portfolio value over time) plus a record of every executed trade,
which Phases 4 (metrics) and 5 (visualisation) consume.

Realism model
-------------
This engine makes a deliberate set of realistic-but-tractable assumptions:

  * Long/flat only — invested (1) or in cash (0). No short selling, which would
    require modelling borrow fees and margin to be honest.
  * Fixed-fraction sizing — deploys `invest_fraction` of the portfolio (default
    95%), keeping a cash buffer so fees never push cash negative.
  * Combined transaction costs — a percentage cost (spread/slippage that scales
    with trade size) plus a flat per-trade fee (commission).
  * Slippage — trades fill at a price slightly worse than the close: buys pay up,
    sells receive less, mimicking real execution.

Lookahead-bias fix
------------------
The Phase 2 Signal on day T is computed from day T's close, which isn't knowable
until the market has closed. Acting on it that day would be lookahead bias. The
engine shifts the signal forward one day (Signal.shift(1)) so a signal generated
on day T is acted on at day T+1's price — the earliest a real trader could react.

Public functions
----------------
run_backtest(df, ...) -> (pd.DataFrame, pd.DataFrame)
    Returns (results, trades): the per-day equity curve and the executed-trade log.

Dependencies: pandas, numpy
"""

import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame,
                 initial_capital: float = 10_000.0,
                 invest_fraction: float = 0.95,
                 cost_pct: float = 0.001,      # 0.1% of trade value
                 flat_fee: float = 1.0,        # $1 per trade
                 slippage_pct: float = 0.0005  # 0.05% price disadvantage
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the strategy day-by-day and return (results, trades).

    Parameters
    ----------
    df : DataFrame with a 'Close' column and a 'Signal' column (from Phase 2).
    initial_capital : starting cash.
    invest_fraction : fraction of the portfolio to deploy when entering a position.
    cost_pct : percentage transaction cost, applied to each trade's value.
    flat_fee : flat cash fee charged on each executed trade.
    slippage_pct : price disadvantage on execution (buys fill higher, sells lower).
    """
    df = df.copy()

    # --- Lookahead fix: act on YESTERDAY's signal at TODAY's price ---
    # A signal computed from day T's close can only be traded from T+1 onward.
    df["Position"] = df["Signal"].shift(1).fillna(0)

    # State variables tracked across the simulation
    cash = initial_capital
    shares = 0.0
    prev_position = 0

    equity_curve = []   # portfolio value each day
    trade_log = []      # one record per executed trade

    for date, row in df.iterrows():
        price = row["Close"]
        target = row["Position"]   # 1 = want to be invested, 0 = want cash

        # --- BUY: we were flat, now we want to be invested ---
        if target == 1 and prev_position == 0:
            fill_price = price * (1 + slippage_pct)      # pay slightly more
            budget = cash * invest_fraction              # keep a buffer
            # Solve for share count so that shares*fill + shares*fill*cost_pct <= budget
            shares_to_buy = budget / (fill_price * (1 + cost_pct))
            trade_value = shares_to_buy * fill_price
            commission = trade_value * cost_pct + flat_fee

            cash -= (trade_value + commission)
            shares += shares_to_buy

            trade_log.append({
                "Date": date, "Type": "BUY", "Price": fill_price,
                "Shares": shares_to_buy, "Cost": commission, "Cash": cash,
            })

        # --- SELL: we were invested, now we want cash ---
        elif target == 0 and prev_position == 1:
            fill_price = price * (1 - slippage_pct)      # receive slightly less
            trade_value = shares * fill_price
            commission = trade_value * cost_pct + flat_fee

            cash += (trade_value - commission)

            trade_log.append({
                "Date": date, "Type": "SELL", "Price": fill_price,
                "Shares": shares, "Cost": commission, "Cash": cash,
            })
            shares = 0.0

        # --- Mark-to-market: portfolio value = cash + current holdings value ---
        holdings_value = shares * price
        total_equity = cash + holdings_value

        equity_curve.append({
            "Date": date, "Close": price, "Cash": cash,
            "Shares": shares, "Holdings": holdings_value, "Equity": total_equity,
        })

        prev_position = target

    results = pd.DataFrame(equity_curve).set_index("Date")
    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        trades = trades.set_index("Date")

    return results, trades


if __name__ == "__main__":
    from data_pipeline import fetch_data, clean_data
    from signals import generate_signals

    data = generate_signals(clean_data(fetch_data("AAPL", period="10y")))
    results, trades = run_backtest(data)

    start_val = 10_000.0
    final_val = results["Equity"].iloc[-1]
    total_return = (final_val / start_val - 1) * 100

    print("--- Backtest summary ---")
    print(f"Initial capital: ${start_val:,.2f}")
    print(f"Final equity:    ${final_val:,.2f}")
    print(f"Total return:    {total_return:+.2f}%")
    print(f"Trades executed: {len(trades)}")
    print(f"Total fees paid: ${trades['Cost'].sum():,.2f}" if not trades.empty else "No trades")

    print("\n--- Executed trades ---")
    print(trades.to_string() if not trades.empty else "None")

    print("\n--- Final 3 days of equity curve ---")
    print(results.tail(3).to_string())
