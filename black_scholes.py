"""
Black-Scholes European option pricer, Greeks, and implied volatility inversion.

Formula reference: Black & Scholes (1973), Hull (2018) Ch. 15-19.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S0, K, r, T, sigma, option_type="call"):
    """
    Black-Scholes price of a European option.

    Parameters
    ----------
    S0 : float - current underlying price
    K : float - strike price
    r : float - risk-free rate (annualized, continuous compounding)
    T : float - time to maturity, in years
    sigma : float - volatility (annualized)
    option_type : "call" or "put"

    Returns
    -------
    float - option price
    """
    if T <= 0 or sigma <= 0:
        # Degenerate case: price collapses to intrinsic value
        intrinsic = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
        return intrinsic

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def bs_greeks(S0, K, r, T, sigma, option_type="call"):
    """
    Compute the standard Black-Scholes Greeks.

    Returns a dict with delta, gamma, vega, theta, rho.
    Theta is per-year; divide by 365 for per-day theta.
    Vega is per unit change in sigma (e.g. per 1.00, not per 1%).
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            -(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(market_price, S0, K, r, T, option_type="call",
                 lo=1e-6, hi=5.0):
    """
    Invert the Black-Scholes formula to find the implied volatility that
    reproduces a given observed market price.

    Uses Brent's method (bracketed root finding) rather than Newton-Raphson
    for robustness -- deep ITM/OTM options can have near-zero vega, which
    makes Newton-Raphson unstable.

    Returns np.nan if no solution is found in [lo, hi].
    """
    def objective(sigma):
        return bs_price(S0, K, r, T, sigma, option_type) - market_price

    try:
        # Check the bracket actually contains a root
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


if __name__ == "__main__":
    # --- Validation against known reference values ---
    # Example from Hull's textbook: S0=42, K=40, r=0.10, T=0.5, sigma=0.20
    # Expected call price approx 4.76
    S0, K, r, T, sigma = 42, 40, 0.10, 0.5, 0.20

    call_price = bs_price(S0, K, r, T, sigma, "call")
    put_price = bs_price(S0, K, r, T, sigma, "put")
    print(f"Call price: {call_price:.4f}  (expected approx 4.76)")
    print(f"Put price:  {put_price:.4f}")

    # Put-call parity check: C - P = S0 - K*exp(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S0 - K * np.exp(-r * T)
    print(f"\nPut-call parity check: {parity_lhs:.6f} vs {parity_rhs:.6f} "
          f"(diff = {abs(parity_lhs - parity_rhs):.2e})")

    greeks = bs_greeks(S0, K, r, T, sigma, "call")
    print("\nGreeks (call):")
    for name, val in greeks.items():
        print(f"  {name}: {val:.5f}")

    # Implied vol inversion round-trip: recover sigma=0.20 from the price
    recovered_sigma = implied_vol(call_price, S0, K, r, T, "call")
    print(f"\nImplied vol recovered: {recovered_sigma:.6f} (expected 0.20000)")
