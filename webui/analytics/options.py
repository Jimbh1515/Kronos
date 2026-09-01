"""
Options analytics: Black-Scholes pricing, Greeks, implied volatility, and
payoff diagrams for common single/multi-leg strategies.
"""
import math

from scipy.stats import norm


def _d1_d2(spot: float, strike: float, t: float, r: float, sigma: float, q: float = 0.0):
    if t <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def black_scholes_price(spot: float, strike: float, t: float, r: float, sigma: float,
                         option_type: str = 'call', q: float = 0.0) -> float:
    """t in years, r/q/sigma as decimals (e.g. 0.05 for 5%)."""
    if t <= 0:
        intrinsic = max(spot - strike, 0) if option_type == 'call' else max(strike - spot, 0)
        return intrinsic
    d1, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    if option_type == 'call':
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def greeks(spot: float, strike: float, t: float, r: float, sigma: float,
           option_type: str = 'call', q: float = 0.0) -> dict:
    if t <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    d1, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (spot * sigma * math.sqrt(t))
    vega = spot * disc_q * pdf_d1 * math.sqrt(t) / 100  # per 1 vol point

    if option_type == 'call':
        delta = disc_q * norm.cdf(d1)
        theta = (-spot * disc_q * pdf_d1 * sigma / (2 * math.sqrt(t))
                 - r * strike * disc_r * norm.cdf(d2)
                 + q * spot * disc_q * norm.cdf(d1)) / 365
        rho = strike * t * disc_r * norm.cdf(d2) / 100
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-spot * disc_q * pdf_d1 * sigma / (2 * math.sqrt(t))
                 + r * strike * disc_r * norm.cdf(-d2)
                 - q * spot * disc_q * norm.cdf(-d1)) / 365
        rho = -strike * t * disc_r * norm.cdf(-d2) / 100

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}


def implied_volatility(market_price: float, spot: float, strike: float, t: float, r: float,
                        option_type: str = 'call', q: float = 0.0,
                        tol: float = 1e-6, max_iter: int = 100) -> float | None:
    """Newton-Raphson with a bisection fallback; returns None if it can't converge."""
    if t <= 0 or market_price <= 0:
        return None

    sigma = 0.3
    for _ in range(max_iter):
        price = black_scholes_price(spot, strike, t, r, sigma, option_type, q)
        vega = greeks(spot, strike, t, r, sigma, option_type, q)['vega'] * 100  # undo /100 scaling
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.01

    # Bisection fallback over a wide vol range.
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        price = black_scholes_price(spot, strike, t, r, mid, option_type, q)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return mid if abs(black_scholes_price(spot, strike, t, r, mid, option_type, q) - market_price) < 0.05 else None


def full_option_analysis(spot: float, strike: float, days_to_expiry: float, r: float, sigma: float,
                          option_type: str = 'call', q: float = 0.0) -> dict:
    t = max(days_to_expiry, 0) / 365
    price = black_scholes_price(spot, strike, t, r, sigma, option_type, q)
    g = greeks(spot, strike, t, r, sigma, option_type, q)
    intrinsic = max(spot - strike, 0) if option_type == 'call' else max(strike - spot, 0)
    return {
        'price': price,
        'intrinsic_value': intrinsic,
        'time_value': price - intrinsic,
        'moneyness': spot / strike,
        **g,
    }


# ---------------------------------------------------------------------------
# Strategy payoff diagrams (at expiry, ignoring time value)
# ---------------------------------------------------------------------------

def _leg_payoff(spot_range, strike, premium, option_type, position):
    """position: +1 long, -1 short."""
    payoffs = []
    for s in spot_range:
        intrinsic = max(s - strike, 0) if option_type == 'call' else max(strike - s, 0)
        payoffs.append(position * (intrinsic - premium))
    return payoffs


STRATEGY_LEGS = {
    'long_call': lambda k, p: [{'strike': k, 'premium': p, 'type': 'call', 'position': 1}],
    'long_put': lambda k, p: [{'strike': k, 'premium': p, 'type': 'put', 'position': 1}],
    'covered_call': lambda k, p: [{'strike': k, 'premium': p, 'type': 'call', 'position': -1, 'is_stock_hedge': True}],
    'protective_put': lambda k, p: [{'strike': k, 'premium': p, 'type': 'put', 'position': 1, 'is_stock_hedge': True}],
    'straddle': lambda k, p: [{'strike': k, 'premium': p, 'type': 'call', 'position': 1}, {'strike': k, 'premium': p, 'type': 'put', 'position': 1}],
}


def strategy_payoff(spot: float, legs: list, spread_pct: float = 0.3, points: int = 61) -> dict:
    """
    legs: list of {strike, premium, type: call/put, position: 1/-1, is_stock_hedge?: bool}.
    is_stock_hedge means this leg is paired with holding/shorting the underlying at `spot`.
    Returns a spot-price grid and the combined P&L curve.
    """
    lo = spot * (1 - spread_pct)
    hi = spot * (1 + spread_pct)
    step = (hi - lo) / (points - 1)
    spot_range = [lo + i * step for i in range(points)]

    total = [0.0] * points
    for leg in legs:
        leg_pnl = _leg_payoff(spot_range, leg['strike'], leg['premium'], leg['type'], leg['position'])
        for i in range(points):
            total[i] += leg_pnl[i]
        if leg.get('is_stock_hedge'):
            stock_dir = -1 if leg['position'] == -1 and leg['type'] == 'call' else 1
            for i in range(points):
                total[i] += stock_dir * (spot_range[i] - spot)

    breakevens = []
    for i in range(1, points):
        if (total[i - 1] < 0) != (total[i] < 0):
            # linear interpolation for the zero crossing
            x0, x1 = spot_range[i - 1], spot_range[i]
            y0, y1 = total[i - 1], total[i]
            if y1 != y0:
                breakevens.append(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))

    return {
        'spot_range': spot_range,
        'pnl': total,
        'max_profit': max(total),
        'max_loss': min(total),
        'breakevens': breakevens,
    }
