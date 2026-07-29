"""
signals.py — Phase 2: Signal Generation
========================================

Translates the momentum hypothesis into concrete, machine-readable trading
signals. This module sits between the data pipeline (Phase 1) and the
backtesting engine (Phase 3): it consumes a clean OHLCV DataFrame and returns
the same frame annotated with moving averages, a daily position signal, and
discrete trade events.

Strategy: Moving Average Crossover
----------------------------------
The momentum hypothesis states that assets in motion tend to stay in motion —
recent winners keep winning, recent losers keep losing. This module operationalises
that idea with the classic 50/200-day moving average crossover:

  * A fast (short-term, default 50-day) moving average tracks recent price action.
  * A slow (long-term, default 200-day) moving average tracks the underlying trend.

When the fast MA sits above the slow MA, recent prices are outpacing the longer
trend — momentum is positive — so the strategy wants to be invested. When the
fast MA drops below the slow MA, momentum has turned, so the strategy exits to
cash. The crossover points themselves are the well-known "Golden Cross" (fast
crosses above slow — bullish) and "Death Cross" (fast crosses below slow — bearish).

Output columns
--------------
generate_signals() adds four columns to the input DataFrame:

  SMA_fast : rolling mean of Close over `fast` days
  SMA_slow : rolling mean of Close over `slow` days
  Signal   : desired position on a given day -> 1 (invested) or 0 (in cash)
  Trade    : change in position -> +1 (buy day), -1 (sell day), 0 (no change)

Signal vs. Trade — why both exist
---------------------------------
`Signal` is the desired *state* every single day (are we in or out?). `Trade` is
the *transition* between states, obtained via Signal.diff(). The distinction
matters downstream: the engine uses `Signal` to know whether the asset is held
on any given day, and `Trade` to know exactly which days incur transaction costs.
Keeping the two separate keeps the simulation logic clean.

Warm-up period
--------------
The slow moving average is undefined for the first `slow` rows (rolling window
not yet full), so SMA_slow is NaN there. The fast > slow comparison evaluates to
False in that region, which correctly defaults the position to 0 (in cash): the
strategy holds nothing until it has enough history to actually compute its rules.

Lookahead bias — IMPORTANT
--------------------------
The Signal on day T is computed from the Close of day T. But in reality today's
close is not known until the market closes, so a trade cannot be executed at that
same price on that same day. Acting on it here would be lookahead bias — the most
common and most dangerous backtesting error, producing results that look great
but are unachievable in practice. This module deliberately does NOT correct for
it; the fix (shifting the signal forward one day so trades fill at the next
available price) is applied in the Phase 3 engine, which is the natural place for
execution-timing logic to live.

Public functions
----------------
generate_signals(df, fast, slow) -> pd.DataFrame
    Add moving averages, the daily position Signal, and discrete Trade events.

Typical usage
-------------
    from data_pipeline import fetch_data, clean_data
    from signals import generate_signals
    data = generate_signals(clean_data(fetch_data("AAPL", period="10y")))

Dependencies: pandas, numpy
"""

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame,
                     fast: int = 50,
                     slow: int = 200) -> pd.DataFrame:
    """Add moving averages, a position signal, and discrete trade events.

    Columns added:
      SMA_fast, SMA_slow : the two moving averages
      Signal             : desired position -> 1 (invested) or 0 (in cash)
      Trade              : change in position -> +1 (buy), -1 (sell), 0 (nothing)
    """
    df = df.copy()

    df["SMA_fast"] = df["Close"].rolling(window=fast).mean()
    df["SMA_slow"] = df["Close"].rolling(window=slow).mean()

    # Desired position: invested whenever the fast MA is above the slow MA.
    df["Signal"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)

    # Before the slow MA has enough data (first `slow` rows), it's NaN and the
    # comparison above defaults to 0 — which is correct (we hold no position
    # until we can actually compute the strategy).

    # Trade = when the position CHANGES. diff() gives +1 on a buy day,
    # -1 on a sell day, 0 otherwise.
    df["Trade"] = df["Signal"].diff().fillna(0)

    return df


if __name__ == "__main__":
    from data_pipeline import fetch_data, clean_data

    data = clean_data(fetch_data("AAPL", period="10y"))
    data = generate_signals(data)

    # Count the discrete trade events
    buys = (data["Trade"] == 1).sum()
    sells = (data["Trade"] == -1).sum()

    print("--- Signal summary ---")
    print(f"Rows with a valid strategy (both MAs present): "
          f"{data['SMA_slow'].notna().sum()}")
    print(f"Days invested (Signal=1): {(data['Signal'] == 1).sum()}")
    print(f"Days in cash  (Signal=0): {(data['Signal'] == 0).sum()}")
    print(f"Buy signals:  {buys}")
    print(f"Sell signals: {sells}")

    print("\n--- Trade events (crossover days) ---")
    events = data[data["Trade"] != 0][["Close", "SMA_fast", "SMA_slow", "Trade"]]
    print(events.to_string())
