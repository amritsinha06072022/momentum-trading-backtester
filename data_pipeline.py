"""
data_pipeline.py — Phase 1: Data Pipeline
==========================================

Fetches, cleans, and caches historical daily OHLCV (Open, High, Low, Close,
Volume) market data for a single asset. This module is the foundation of the
Momentum Trading Backtester: every later phase (signal generation, simulation,
performance metrics) consumes the clean DataFrame produced here.

Data source
-----------
Price data is pulled from Yahoo Finance via the `yfinance` library. Prices are
requested with auto-adjustment enabled, which back-adjusts the historical series
for stock splits and dividends. This is essential for a backtest: without it, a
split (e.g. Apple's 4-for-1 in 2020) would appear as a sudden price crash and
generate false trading signals.

Caching & reproducibility
--------------------------
The first fetch for a given (ticker, period) is saved to `data/<ticker>_<period>.csv`,
and subsequent runs load from that cache instead of re-hitting the API. This
serves two purposes:
  1. Speed — no network round-trip on every run.
  2. Reproducibility — backtests run against a fixed, frozen dataset, so results
     are comparable across strategy tweaks rather than shifting as live data
     updates. Delete the CSV (or pass use_cache=False) to force a fresh download.

The `data/` directory is created relative to this file, so the script behaves
identically regardless of the current working directory, and is git-ignored
since it is fully regenerable.

Public functions
----------------
fetch_data(ticker, period, use_cache) -> pd.DataFrame
    Download (or load from cache) the raw OHLCV series for a ticker.

clean_data(df) -> pd.DataFrame
    Trim to core OHLCV columns, enforce a sorted, de-duplicated DateTime index,
    forward-fill small gaps, and drop leading NaNs. Returns analysis-ready data.

Typical usage
-------------
    from data_pipeline import fetch_data, clean_data
    data = clean_data(fetch_data("AAPL", period="10y"))

Dependencies: pandas, yfinance
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

# data/ folder lives next to THIS file, regardless of the working directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_data(ticker: str, period: str = "10y", use_cache: bool = True) -> pd.DataFrame:
    """Download daily OHLCV data for `ticker`, caching to CSV so we don't
    hammer the API on every run."""
    cache_file = DATA_DIR / f"{ticker}_{period}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading {ticker} from cache: {cache_file.name}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print(f"Downloading {ticker} ({period}) from Yahoo Finance...")
    # auto_adjust=True automatically adjusts prices for stock splits and
    # dividends — that handles two of the "messy data" problems for you.
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")

    # yfinance returns a timezone-aware index; strip it so the CSV round-trips cleanly
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.to_csv(cache_file)
    print(f"Saved {len(df)} rows to {cache_file.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to core OHLCV, handle gaps, and guarantee a sorted DateTime index."""
    df = df.copy()

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df[~df.index.duplicated(keep="first")]  # drop duplicate dates
    df = df.ffill()                              # forward-fill small gaps
    df = df.dropna(subset=["Close"])             # drop any leading NaNs

    return df


if __name__ == "__main__":
    TICKER = "AAPL"  # swap for "BTC-USD", "MSFT", etc.

    raw = fetch_data(TICKER, period="10y")
    data = clean_data(raw)

    print("\n--- Data summary ---")
    print(f"Ticker:      {TICKER}")
    print(f"Rows:        {len(data)}")
    print(f"Date range:  {data.index.min().date()} -> {data.index.max().date()}")
    print(f"Columns:     {list(data.columns)}")
    print("\nFirst 3 rows:")
    print(data.head(3))
    print("\nMissing values per column:")
    print(data.isna().sum())
