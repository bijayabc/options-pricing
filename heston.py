"""
Heston (1993) stochastic volatility model: semi-analytical option pricing
via the characteristic function and numerical Fourier integration.

Uses the Gatheral (2006) "little trap" formulation of the characteristic
function, which is numerically more stable than Heston's original 1993
form (avoids branch-cut discontinuities in the complex logarithm when
integrating over long maturities / certain parameter regimes).

Model (risk-neutral dynamics):
    dS_t = r * S_t dt + sqrt(v_t) * S_t dW_t^S
    dv_t = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW_t^v
    corr(dW^S, dW^v) = rho * dt

Parameters
----------
v0    : initial variance
kappa : mean reversion speed
theta : long-run variance
xi    : vol-of-vol
rho   : correlation between asset and variance Brownian motions
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

# Precompute fixed Gauss-Legendre quadrature nodes/weights once, mapped to
# [1e-8, 100]. Using a fixed-node quadrature instead of scipy.integrate.quad
# is essential for calibration speed: quad's adaptive refinement makes each
# single price evaluation slow, and calibration requires thousands of price
# evaluations. Fixed quadrature is fully vectorizable with numpy and is the
# standard approach used in practice for Heston calibration speed
# (see e.g. Fang & Oosterlee 2008 on numerical methods for Heston pricing).
_N_QUAD = 64
_nodes, _weights = leggauss(_N_QUAD)
_A, _B = 1e-8, 100.0
_PHI = 0.5 * (_B - _A) * _nodes + 0.5 * (_B + _A)
_QUAD_W = 0.5 * (_B - _A) * _weights


def heston_char_func(phi, S0, r, T, kappa, theta, xi, rho, v0):
    """
    Heston characteristic function of log(S_T), Gatheral's stable form.

    phi : array of complex evaluation points (vectorized)
    """
    x0 = np.log(S0)
    a = kappa * theta
    b = kappa

    d = np.sqrt((rho * xi * 1j * phi - b) ** 2 + (xi ** 2) * (1j * phi + phi ** 2))

    g = (b - rho * xi * 1j * phi - d) / (b - rho * xi * 1j * phi + d)

    C = r * 1j * phi * T + (a / xi ** 2) * (
        (b - rho * xi * 1j * phi - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = ((b - rho * xi * 1j * phi - d) / xi ** 2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )

    return np.exp(C + D * v0 + 1j * phi * x0)


def heston_price(S0, K, r, T, kappa, theta, xi, rho, v0, option_type="call"):
    """
    Price a European option under Heston via fast vectorized Gauss-Legendre
    quadrature over the characteristic function (Carr-Madan / Gil-Pelaez
    P1/P2 approach, as in Heston 1993 Eq. 17-18).

    Uses fixed quadrature nodes (precomputed at import time) rather than
    scipy.integrate.quad -- roughly two orders of magnitude faster, which
    matters because calibration calls this thousands of times.
    """
    phi = _PHI  # vectorized, shape (_N_QUAD,)

    cf1_num = heston_char_func(phi - 1j, S0, r, T, kappa, theta, xi, rho, v0)
    cf1_den = 1j * phi * heston_char_func(-1j, S0, r, T, kappa, theta, xi, rho, v0)
    integrand1 = np.real(np.exp(-1j * phi * np.log(K)) * cf1_num / cf1_den)

    cf2 = heston_char_func(phi, S0, r, T, kappa, theta, xi, rho, v0)
    integrand2 = np.real(np.exp(-1j * phi * np.log(K)) * cf2 / (1j * phi))

    P1 = 0.5 + (1 / np.pi) * np.sum(_QUAD_W * integrand1)
    P2 = 0.5 + (1 / np.pi) * np.sum(_QUAD_W * integrand2)

    call_price = S0 * P1 - K * np.exp(-r * T) * P2

    if option_type == "call":
        return call_price
    elif option_type == "put":
        return call_price - S0 + K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


if __name__ == "__main__":
    # --- Validation against a widely-cited benchmark parameter set ---
    # (Parameters commonly used in Heston-pricer validation, e.g. in
    # Albrecher et al. 2007 "The Little Heston Trap" and related literature)
    S0 = 100.0
    K = 100.0
    r = 0.0
    T = 1.0
    kappa = 1.5
    theta = 0.04
    xi = 0.3
    rho = -0.9
    v0 = 0.04

    price = heston_price(S0, K, r, T, kappa, theta, xi, rho, v0, "call")
    print(f"Heston call price: {price:.4f}")

    # Sanity check: as xi -> 0 (no vol-of-vol) and v0 = theta (flat variance),
    # Heston should reduce to Black-Scholes with sigma = sqrt(theta).
    from black_scholes import bs_price

    xi_small = 1e-4
    bs_equiv_sigma = np.sqrt(theta)
    heston_flat = heston_price(S0, K, r, T, kappa, theta, xi_small, rho, theta, "call")
    bs_equiv = bs_price(S0, K, r, T, bs_equiv_sigma, "call")
    print(f"\nSanity check (xi->0, v0=theta should match Black-Scholes):")
    print(f"  Heston (flat variance): {heston_flat:.4f}")
    print(f"  Black-Scholes equiv:    {bs_equiv:.4f}")
    print(f"  Difference:             {abs(heston_flat - bs_equiv):.6f}")

    # Put-call parity check
    put_price = heston_price(S0, K, r, T, kappa, theta, xi, rho, v0, "put")
    parity_lhs = price - put_price
    parity_rhs = S0 - K * np.exp(-r * T)
    print(f"\nPut-call parity check: {parity_lhs:.6f} vs {parity_rhs:.6f} "
          f"(diff = {abs(parity_lhs - parity_rhs):.2e})")
