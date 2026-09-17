# Options Pricing Under Stochastic Volatility: Black-Scholes vs. Heston

A comparative analysis of the Black-Scholes and Heston models, calibrated against real SPY options market data.

## Overview

This project investigates when stochastic volatility modeling (Heston) provides meaningfully better options pricing accuracy than the constant-volatility assumption underlying Black-Scholes. Both models are implemented from their underlying mathematical formulas, validated against known theoretical relationships, and calibrated to real market data across a range of strikes and maturities.

**Research question:** Under what market conditions does stochastic volatility modeling yield materially superior pricing performance, and how do Heston's calibrated parameters behave once fit to real data?

## Files

| File | Description |
|---|---|
| `black_scholes.py` | Closed-form Black-Scholes pricer, Greeks, and implied volatility inversion |
| `heston.py` | Heston stochastic volatility pricer via characteristic function / Fourier integration |
| `fetch_spx_data.py` | Fetches and cleans real SPY options chain data via `yfinance` |
| `calibrate_heston.py` | Calibrates Heston's parameters to market data and compares pricing accuracy against Black-Scholes |

## Pipeline

```
fetch_spx_data.py  -->  spy_options_clean.csv
                              |
calibrate_heston.py  (imports black_scholes.py + heston.py)
                              |
                    model_comparison_results.csv
```

## Setup

```bash
pip install numpy scipy pandas yfinance
```

## Usage

```bash
# 1. Fetch and clean real SPY options data
python fetch_spx_data.py

# 2. Calibrate Heston and compare against Black-Scholes
python calibrate_heston.py
```

`fetch_spx_data.py` requires normal internet access (it calls the Yahoo Finance API) and produces `spy_options_clean.csv`. `calibrate_heston.py` reads that file and produces `model_comparison_results.csv`.

## Methodology

### Black-Scholes (`black_scholes.py`)
Standard closed-form European option pricing formula, assuming constant volatility. Implied volatility is recovered via Brent's method root-finding (chosen over Newton-Raphson for robustness with near-zero vega at deep ITM/OTM strikes).

### Heston (`heston.py`)
Models volatility itself as a mean-reverting stochastic process, correlated with the asset price (capturing the empirically observed "leverage effect"). Since Heston has no simple closed-form price, it is priced via numerical integration of its characteristic function, using Gatheral's "little trap" formulation for numerical stability, and fast vectorized Gauss-Legendre quadrature for computational speed.

**Validation:** both pricers are checked against put-call parity (must hold exactly for any correct implementation), and Heston is confirmed to converge to Black-Scholes when vol-of-vol → 0 and initial variance = long-run variance.

### Data (`fetch_spx_data.py`)
Real SPY option chains pulled via `yfinance`, spanning expirations from ~3 weeks to ~1 year to maturity (deliberately spread across the term structure rather than clustering near-term). Cleaned by removing illiquid quotes, excessively wide bid-ask spreads, and near-zero prices.

### Calibration (`calibrate_heston.py`)
Heston's five parameters (κ: mean-reversion speed, θ: long-run variance, ξ: vol-of-vol, ρ: price-variance correlation, v₀: initial variance) are calibrated via Nelder-Mead optimization with penalty-based constraints, restricted to a moneyness band (0.7–1.3) and out-of-the-money contracts, where option prices are most informative about volatility. Black-Scholes is calibrated to a single constant σ across the same dataset for a fair, non-circular comparison. Accuracy is reported via RMSE and MAPE, both overall and by maturity bucket.

## Key Findings

- Heston reduces pricing RMSE relative to Black-Scholes across all maturity buckets tested, though the magnitude of improvement varies across data snapshots (observed roughly 28-66% RMSE reduction across two independent calibration runs)
- The calibrated correlation parameter ρ repeatedly hits the mathematical boundary of -1.0, suggesting the real market's volatility skew may be steeper than a pure-diffusion model like Heston can fully capture — pointing toward jump-diffusion extensions (e.g., the Bates model) as a natural next step
- Heston calibration is prone to local optima (a degenerate "flat volatility" solution with ξ ≈ 0), a known and documented challenge in the literature; multiple random restarts are used to mitigate this

## Limitations

- MAPE is inflated by cheap, deep out-of-the-money contracts where small dollar errors translate to large percentage errors; RMSE is treated as the primary accuracy metric
- Data reflects real-time market snapshots rather than a historical panel, limiting the ability to test performance across distinct volatility regimes (e.g., high- vs. low-VIX periods) as originally proposed
- Calibrated parameters (e.g., very high κ in some runs) can be economically implausible, likely reflecting local-optima and identifiability challenges rather than a specification error

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
