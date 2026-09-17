"""
Calibrate the Heston model to real SPY options market data, and compare
Black-Scholes vs. Heston pricing accuracy across moneyness and maturity.

Run this AFTER fetch_spx_data.py has produced spy_options_clean.csv.

    python calibrate_heston.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm

from black_scholes import bs_price, implied_vol
from heston import heston_price

RISK_FREE_RATE = 0.045  # approx short-term T-bill rate; adjust as needed


def load_and_filter_data(path="spy_options_clean.csv",
                          moneyness_lo=0.7, moneyness_hi=1.3):
    """
    Load the cleaned options data and restrict to a moneyness band where
    prices are actually informative about volatility (deep ITM/OTM prices
    are dominated by intrinsic value and contribute little to calibration,
    while adding numerical noise).
    """
    df = pd.read_csv(path)
    n0 = len(df)
    df = df[(df["moneyness"] >= moneyness_lo) & (df["moneyness"] <= moneyness_hi)]

    # Use out-of-the-money options for each side (standard practice: OTM
    # options are more liquid and avoid early-exercise / put-call
    # asymmetries that complicate American-style SPY options).
    calls = df[(df["option_type"] == "call") & (df["strike"] >= df["S0"])]
    puts = df[(df["option_type"] == "put") & (df["strike"] < df["S0"])]
    df = pd.concat([calls, puts], ignore_index=True)

    print(f"Filtered to moneyness [{moneyness_lo}, {moneyness_hi}], OTM only: "
          f"{n0} -> {len(df)} rows")
    return df


def compute_implied_vols(df, r=RISK_FREE_RATE):
    """Add a market implied-vol column via Black-Scholes inversion."""
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(row["mid_price"], row["S0"], row["strike"],
                          r, row["T"], row["option_type"])
        ivs.append(iv)
    df = df.copy()
    df["implied_vol"] = ivs
    n_before = len(df)
    df = df.dropna(subset=["implied_vol"])
    print(f"Implied vol inversion succeeded for {len(df)}/{n_before} contracts")
    return df


def heston_objective(params, df, r=RISK_FREE_RATE):
    """
    Sum of squared pricing errors between Heston model prices and
    observed market mid-prices, across the calibration dataset.

    Uses a penalty (rather than hard optimizer bounds) to keep parameters
    in economically/mathematically valid territory. This is more robust
    for Nelder-Mead than scipy's `bounds` argument, which was found to
    degrade convergence significantly (the simplex has less room to
    reflect/expand near a boundary). The penalty approach lets the
    optimizer search freely while still being pushed back from invalid
    regions like |rho| > 1.

    Note: kappa, theta, xi, v0 are conventionally kept non-negative by
    convention (xi represents a magnitude -- "vol of vol"), with rho
    carrying the sign of the price-variance relationship. The model has
    a sign symmetry (xi -> -xi, rho -> -rho leaves prices unchanged,
    since they only ever appear as the product rho*xi or as xi^2) --
    fixing xi >= 0 removes this redundancy.
    """
    kappa, theta, xi, rho, v0 = params

    penalty = 0.0
    for val, name in [(kappa, "kappa"), (theta, "theta"), (xi, "xi"), (v0, "v0")]:
        if val <= 0:
            penalty += 1e6 * (abs(val) + 0.01)
    if abs(rho) > 1:
        penalty += 1e6 * (abs(rho) - 1)

    if penalty > 0:
        return 1e6 + penalty  # don't even bother pricing in invalid regions

    # Feller condition soft penalty: 2*kappa*theta > xi^2 keeps variance
    # away from zero. Not strictly required for the pricer to run, but
    # violations often signal an unstable/nonsensical parameter region.
    if 2 * kappa * theta < xi ** 2:
        penalty += 1e3 * (xi ** 2 - 2 * kappa * theta)

    sq_errors = []
    for _, row in df.iterrows():
        try:
            model_price = heston_price(
                row["S0"], row["strike"], r, row["T"],
                kappa, theta, xi, rho, v0, row["option_type"]
            )
            if not np.isfinite(model_price):
                continue
            sq_errors.append((model_price - row["mid_price"]) ** 2)
        except Exception:
            continue

    if not sq_errors:
        return 1e10

    return np.mean(sq_errors) + penalty


def calibrate_heston(df, r=RISK_FREE_RATE, n_restarts=5):
    """
    Calibrate Heston's 5 parameters via numerical optimization
    (Nelder-Mead, since the objective is not smooth/differentiable
    in a form scipy can exploit gradients for easily).

    Uses several random restarts, unconstrained (validity enforced via
    penalty inside the objective -- see heston_objective), to reduce the
    chance of landing in a poor local minimum -- a known risk with Heston
    calibration.
    """
    init_ranges = {
        "kappa": (0.5, 5.0),
        "theta": (0.01, 0.15),
        "xi":    (0.1, 1.2),
        "rho":   (-0.9, 0.9),
        "v0":    (0.01, 0.15),
    }

    rng = np.random.default_rng(42)
    best_result = None
    best_obj = np.inf

    for attempt in range(n_restarts):
        x0 = np.array([
            rng.uniform(*init_ranges["kappa"]),
            rng.uniform(*init_ranges["theta"]),
            rng.uniform(*init_ranges["xi"]),
            rng.uniform(*init_ranges["rho"]),
            rng.uniform(*init_ranges["v0"]),
        ])

        result = minimize(
            heston_objective, x0, args=(df, r),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
        )

        print(f"  Restart {attempt+1}: objective = {result.fun:.6f}, "
              f"params = {np.round(result.x, 4)}")

        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    kappa, theta, xi, rho, v0 = best_result.x
    print(f"\nBest calibration found (objective = {best_obj:.6f}):")
    print(f"  kappa = {kappa:.4f}  (mean reversion speed)")
    print(f"  theta = {theta:.4f}  (long-run variance, sqrt = {np.sqrt(theta):.4f})")
    print(f"  xi    = {xi:.4f}  (vol-of-vol)")
    print(f"  rho   = {rho:.4f}  (correlation -- expect negative for equities)")
    print(f"  v0    = {v0:.4f}  (initial variance, sqrt = {np.sqrt(v0):.4f})")

    feller = 2 * kappa * theta - xi ** 2
    print(f"  Feller condition (2*kappa*theta - xi^2): {feller:.4f} "
          f"({'satisfied' if feller > 0 else 'VIOLATED -- variance can hit zero'})")

    return {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}


def calibrate_single_bs_sigma(df, r=RISK_FREE_RATE):
    """
    Fit a single constant sigma for Black-Scholes across the whole dataset,
    by minimizing squared pricing error -- analogous to how Heston's 5
    parameters are calibrated.

    NOTE: using each option's own implied volatility as BS's sigma would
    make BS trivially reproduce every price exactly (implied vol is *defined*
    as the sigma that makes BS match the market price), which is a circular,
    meaningless comparison. A fair test of "does constant volatility work"
    requires one shared sigma, since that is what Black-Scholes actually
    assumes.
    """
    def objective(sigma):
        sigma = sigma[0]
        errors = [
            bs_price(row["S0"], row["strike"], r, row["T"], sigma, row["option_type"])
            - row["mid_price"]
            for _, row in df.iterrows()
        ]
        return np.mean(np.array(errors) ** 2)

    result = minimize(objective, x0=[0.2], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8})
    sigma_fit = result.x[0]
    print(f"Fitted single BS sigma: {sigma_fit:.4f}")
    return sigma_fit


def compare_models(df, heston_params, bs_sigma, r=RISK_FREE_RATE):
    """
    Compute BS (single fitted sigma) and Heston prices for every contract,
    and report RMSE / MAPE for each model overall and sliced by maturity
    bucket.
    """
    df = df.copy()
    bs_prices, heston_prices = [], []

    for _, row in df.iterrows():
        bsp = bs_price(row["S0"], row["strike"], r, row["T"],
                        bs_sigma, row["option_type"])
        hp = heston_price(row["S0"], row["strike"], r, row["T"],
                           heston_params["kappa"], heston_params["theta"],
                           heston_params["xi"], heston_params["rho"],
                           heston_params["v0"], row["option_type"])
        bs_prices.append(bsp)
        heston_prices.append(hp)

    df["bs_price"] = bs_prices
    df["heston_price"] = heston_prices
    df["bs_error"] = df["bs_price"] - df["mid_price"]
    df["heston_error"] = df["heston_price"] - df["mid_price"]

    def rmse(x):
        return np.sqrt(np.mean(x ** 2))

    def mape(err, actual):
        return np.mean(np.abs(err) / actual) * 100

    print("\n=== Overall pricing accuracy ===")
    print(f"Black-Scholes: RMSE = {rmse(df['bs_error']):.4f}, "
          f"MAPE = {mape(df['bs_error'], df['mid_price']):.2f}%")
    print(f"Heston:        RMSE = {rmse(df['heston_error']):.4f}, "
          f"MAPE = {mape(df['heston_error'], df['mid_price']):.2f}%")

    # Bucket by maturity
    df["maturity_bucket"] = pd.cut(
        df["T"], bins=[0, 0.1, 0.3, 0.6, 2.0],
        labels=["<1mo", "1-4mo", "4-7mo", ">7mo"]
    )

    print("\n=== By maturity bucket ===")
    for bucket, group in df.groupby("maturity_bucket", observed=True):
        if len(group) == 0:
            continue
        print(f"{bucket} (n={len(group)}): "
              f"BS RMSE={rmse(group['bs_error']):.4f}, "
              f"Heston RMSE={rmse(group['heston_error']):.4f}")

    return df


if __name__ == "__main__":
    print("Loading and filtering data...")
    df = load_and_filter_data()

    print("\nComputing market implied volatilities...")
    df = compute_implied_vols(df)

    print("\nCalibrating Heston model (this may take a few minutes)...")
    heston_params = calibrate_heston(df)

    print("\nFitting single constant sigma for Black-Scholes comparison...")
    bs_sigma = calibrate_single_bs_sigma(df)

    print("\nComparing model accuracy...")
    results_df = compare_models(df, heston_params, bs_sigma)

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved full results to model_comparison_results.csv")
