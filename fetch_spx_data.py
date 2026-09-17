"""
Fetch and clean SPY options chain data via yfinance.

NOTE: This script must be run in an environment with normal internet
access (your own machine) -- it calls the Yahoo Finance API, which is
not reachable from a sandboxed tool environment.

Usage:
    pip install yfinance
    python fetch_spx_data.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def select_spread_expirations(expirations, today, n_target=8):
    """
    Select expirations spread across the term structure (short, medium,
    and longer maturities) rather than just the first N chronologically.

    SPY has very frequent (near-daily/weekly) near-term expirations, so
    naively taking "the first N" bunches everything into the next 1-2
    weeks and leaves no medium/long-dated coverage -- which defeats the
    purpose of comparing BS vs. Heston across maturities.

    Strategy: bucket available expirations into rough maturity bands and
    pick a representative expiration from each band, so the final sample
    spans days -> weeks -> months -> (if available) a year+.
    """
    exp_dates = [pd.Timestamp(e) for e in expirations]
    days_out = [(d - today).days for d in exp_dates]

    # Target maturity bands (in days), roughly log-spaced
    target_days = [7, 14, 30, 60, 90, 180, 270, 365]
    target_days = target_days[:n_target]

    chosen = []
    used = set()
    for target in target_days:
        # find the available expiration closest to this target
        diffs = [abs(d - target) for d in days_out]
        idx = int(np.argmin(diffs))
        if expirations[idx] not in used and days_out[idx] > 0:
            chosen.append(expirations[idx])
            used.add(expirations[idx])

    return sorted(chosen, key=lambda e: pd.Timestamp(e))


def fetch_option_chains(ticker_symbol="SPY", max_expirations=8):
    """
    Pull option chains (calls and puts) across multiple expirations
    for the given ticker, spread across short/medium/long maturities.

    Returns a single DataFrame with one row per option contract,
    including a computed time-to-maturity column.
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    if not expirations:
        raise RuntimeError(
            f"No option expirations returned for {ticker_symbol}. "
            "Check ticker symbol and internet connectivity."
        )

    print(f"Found {len(expirations)} total expirations for {ticker_symbol}.")
    today = pd.Timestamp.today().normalize()
    expirations_to_use = select_spread_expirations(expirations, today, max_expirations)
    print(f"Selected {len(expirations_to_use)} expirations spanning the term structure: "
          f"{expirations_to_use}")

    # Current spot price
    S0 = ticker.history(period="1d")["Close"].iloc[-1]
    print(f"Current {ticker_symbol} price: {S0:.2f}")

    all_rows = []
    for exp_str in expirations_to_use:
        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            print(f"  Skipping {exp_str}: fetch failed ({e})")
            continue

        exp_date = pd.Timestamp(exp_str)
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue

        for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df.copy()
            df["option_type"] = opt_type
            df["expiration"] = exp_str
            df["T"] = T
            df["S0"] = S0
            all_rows.append(df)

        print(f"  {exp_str}: T={T:.3f}y, "
              f"{len(chain.calls)} calls, {len(chain.puts)} puts")

    full_df = pd.concat(all_rows, ignore_index=True)
    return full_df, S0


def clean_option_data(df, min_volume=1, max_spread_pct=0.5, min_price=0.05):
    """
    Clean the raw options data:
      - drop rows with no bid/ask (illiquid / stale)
      - drop rows with excessively wide bid-ask spreads (relative to mid)
      - drop rows with near-zero option price (numerically unstable for
        implied-vol inversion and calibration)
      - drop obvious data errors (bid > ask)

    Returns the cleaned DataFrame with a 'mid_price' column added.
    """
    n0 = len(df)

    df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
    df = df[df["bid"] <= df["ask"]]

    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df = df[df["mid_price"] >= min_price]

    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid_price"]
    df = df[df["spread_pct"] <= max_spread_pct]

    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= min_volume]

    # Moneyness, useful for later slicing of results by ITM/ATM/OTM
    df["moneyness"] = df["strike"] / df["S0"]

    n1 = len(df)
    print(f"\nCleaning: {n0} rows -> {n1} rows "
          f"({n0 - n1} dropped, {100*(n0-n1)/n0:.1f}%)")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw_df, S0 = fetch_option_chains("SPY", max_expirations=8)
    clean_df = clean_option_data(raw_df)

    print("\nCleaned data summary:")
    print(clean_df[["expiration", "T", "strike", "option_type",
                     "bid", "ask", "mid_price", "moneyness"]].describe(include="all"))

    out_path = "spy_options_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned data to {out_path}")
    print(f"Total contracts: {len(clean_df)}")
    print(f"Expirations covered: {clean_df['expiration'].nunique()}")
    print(f"Strike range: {clean_df['strike'].min():.0f} - {clean_df['strike'].max():.0f}")
