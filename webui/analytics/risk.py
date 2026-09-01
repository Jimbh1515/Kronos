"""
Portfolio / single-asset risk analytics: returns, volatility, drawdown,
risk-adjusted ratios, Value-at-Risk, and correlation.
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or cumulative <= 0:
        return 0.0
    return float(cumulative ** (1 / n_years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    excess = returns - risk_free_rate / periods_per_year
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_dev = downside.std(ddof=0)
    if downside_dev == 0 or np.isnan(downside_dev):
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> dict:
    cum_max = prices.cummax()
    drawdown = prices / cum_max - 1
    trough_idx = drawdown.idxmin()
    max_dd = float(drawdown.min())
    peak_idx = prices.loc[:trough_idx].idxmax() if trough_idx is not None else None
    # recovery: first index after trough where price regains the prior peak
    recovery_idx = None
    if trough_idx is not None:
        peak_val = prices.loc[peak_idx]
        after = prices.loc[trough_idx:]
        recovered = after[after >= peak_val]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]
    return {
        'max_drawdown_pct': max_dd * 100,
        'peak_index': str(peak_idx) if peak_idx is not None else None,
        'trough_index': str(trough_idx) if trough_idx is not None else None,
        'recovery_index': str(recovery_idx) if recovery_idx is not None else None,
        'drawdown_series': drawdown,
    }


def calmar_ratio(returns: pd.Series, prices: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    ar = annualized_return(returns, periods_per_year)
    dd = max_drawdown(prices)['max_drawdown_pct'] / 100
    if dd == 0:
        return 0.0
    return float(ar / abs(dd))


def value_at_risk(returns: pd.Series, confidence: float = 0.95, method: str = 'historical') -> float:
    """Single-period VaR, expressed as a positive percentage loss."""
    if len(returns) == 0:
        return 0.0
    if method == 'parametric':
        from scipy.stats import norm
        mu, sigma = returns.mean(), returns.std(ddof=0)
        z = norm.ppf(1 - confidence)
        var = -(mu + z * sigma)
    else:
        var = -np.percentile(returns, (1 - confidence) * 100)
    return float(max(var, 0.0) * 100)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR): average loss beyond the VaR threshold."""
    if len(returns) == 0:
        return 0.0
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean() * 100)


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return 0.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    var = cov[1, 1]
    if var == 0:
        return 0.0
    return float(cov[0, 1] / var)


def correlation_matrix(price_dict: dict) -> pd.DataFrame:
    """price_dict: {symbol: close-price Series}. Returns correlation of daily returns."""
    rets = {sym: simple_returns(series) for sym, series in price_dict.items()}
    df = pd.DataFrame(rets)
    return df.corr()


def rolling_volatility(returns: pd.Series, window: int = 21, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    return returns.rolling(window).std(ddof=0) * np.sqrt(periods_per_year) * 100


def risk_summary(prices: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                  benchmark_prices: pd.Series | None = None) -> dict:
    """One-shot risk report for a single price series."""
    rets = simple_returns(prices)
    dd = max_drawdown(prices)
    summary = {
        'total_return_pct': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100) if len(prices) > 1 else 0.0,
        'annualized_return_pct': annualized_return(rets, periods_per_year) * 100,
        'annualized_volatility_pct': annualized_volatility(rets, periods_per_year) * 100,
        'sharpe_ratio': sharpe_ratio(rets, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(rets, risk_free_rate, periods_per_year),
        'calmar_ratio': calmar_ratio(rets, prices, periods_per_year),
        'max_drawdown_pct': dd['max_drawdown_pct'],
        'drawdown_peak': dd['peak_index'],
        'drawdown_trough': dd['trough_index'],
        'drawdown_recovery': dd['recovery_index'],
        'var_95_historical_pct': value_at_risk(rets, 0.95, 'historical'),
        'var_95_parametric_pct': value_at_risk(rets, 0.95, 'parametric'),
        'cvar_95_pct': conditional_var(rets, 0.95),
        'skewness': float(rets.skew()) if len(rets) > 2 else 0.0,
        'kurtosis': float(rets.kurtosis()) if len(rets) > 3 else 0.0,
        'best_day_pct': float(rets.max() * 100) if len(rets) else 0.0,
        'worst_day_pct': float(rets.min() * 100) if len(rets) else 0.0,
        'positive_days_pct': float((rets > 0).mean() * 100) if len(rets) else 0.0,
    }
    if benchmark_prices is not None:
        bench_rets = simple_returns(benchmark_prices)
        summary['beta'] = beta(rets, bench_rets)
    return summary
