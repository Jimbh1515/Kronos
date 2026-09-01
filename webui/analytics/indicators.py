"""
Technical analysis indicator library, implemented directly on pandas/numpy so it has
no compiled dependencies (no TA-Lib) and stays portable across platforms.

Every function takes a DataFrame with at least ``open, high, low, close`` (and
``volume`` where needed) columns and returns either a Series or a DataFrame of the
indicator's output(s), aligned to the input index.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, length: int = 20) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int = 20) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wma(series: pd.Series, length: int = 20) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100).where(avg_loss != 0, 100).mask(avg_gain == 0, 0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'hist': hist})


def stochastic(df: pd.DataFrame, k_length: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    low_min = df['low'].rolling(k_length).min()
    high_max = df['high'].rolling(k_length).max()
    raw_k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return pd.DataFrame({'k': k, 'd': d})


def roc(series: pd.Series, length: int = 12) -> pd.Series:
    return (series / series.shift(length) - 1) * 100


def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_max = df['high'].rolling(length).max()
    low_min = df['low'].rolling(length).min()
    return -100 * (high_max - df['close']) / (high_max - low_min).replace(0, np.nan)


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(length).mean()
    mean_dev = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    money_flow = tp * df['volume']
    delta = tp.diff()
    pos_flow = money_flow.where(delta > 0, 0.0).rolling(length).sum()
    neg_flow = money_flow.where(delta < 0, 0.0).rolling(length).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


# ---------------------------------------------------------------------------
# Volatility / bands
# ---------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    ranges = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def bollinger_bands(series: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, length)
    std = series.rolling(length).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({'mid': mid, 'upper': upper, 'lower': lower, 'bandwidth': bandwidth, 'percent_b': percent_b})


def keltner_channels(df: pd.DataFrame, length: int = 20, atr_length: int = 10, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df['close'], length)
    band = atr(df, atr_length) * mult
    return pd.DataFrame({'mid': mid, 'upper': mid + band, 'lower': mid - band})


def donchian_channels(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    upper = df['high'].rolling(length).max()
    lower = df['low'].rolling(length).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({'upper': upper, 'mid': mid, 'lower': lower})


def historical_volatility(series: pd.Series, length: int = 20, periods_per_year: int = 252) -> pd.Series:
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(length).std(ddof=0) * np.sqrt(periods_per_year) * 100


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return pd.DataFrame({'adx': adx_val, 'plus_di': plus_di, 'minus_di': minus_di})


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high, low = df['high'].values, df['low'].values
    n = len(df)
    sar = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=df.index)
    bull = True
    af = step
    ep = low[0]
    sar[0] = high[0]
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = step
    return pd.Series(sar, index=df.index)


def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52, displacement: int = 26) -> pd.DataFrame:
    conv_line = (df['high'].rolling(conv).max() + df['low'].rolling(conv).min()) / 2
    base_line = (df['high'].rolling(base).max() + df['low'].rolling(base).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(displacement)
    span_b_line = ((df['high'].rolling(span_b).max() + df['low'].rolling(span_b).min()) / 2).shift(displacement)
    lagging_span = df['close'].shift(-displacement)
    return pd.DataFrame({
        'conversion': conv_line,
        'base': base_line,
        'span_a': span_a,
        'span_b': span_b_line,
        'lagging_span': lagging_span,
    })


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-agnostic cumulative VWAP over the full input (resets only at the start)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].cumsum()
    cum_vol_price = (tp * df['volume']).cumsum()
    return cum_vol_price / cum_vol.replace(0, np.nan)


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution line."""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    return (clv.fillna(0) * df['volume']).cumsum()


# ---------------------------------------------------------------------------
# Support / resistance helpers
# ---------------------------------------------------------------------------

def pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Classic floor-trader pivot points from the prior period's OHLC."""
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}


def fibonacci_retracement(high: float, low: float) -> dict:
    diff = high - low
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    return {f'{lvl:.3f}': high - diff * lvl for lvl in levels}


# ---------------------------------------------------------------------------
# Registry: name -> callable metadata, used by the web API / screener
# ---------------------------------------------------------------------------

INDICATOR_SPECS = {
    'sma': {'fn': lambda df, **p: sma(df['close'], p.get('length', 20)), 'overlay': True, 'params': {'length': 20}},
    'ema': {'fn': lambda df, **p: ema(df['close'], p.get('length', 20)), 'overlay': True, 'params': {'length': 20}},
    'wma': {'fn': lambda df, **p: wma(df['close'], p.get('length', 20)), 'overlay': True, 'params': {'length': 20}},
    'bollinger': {'fn': lambda df, **p: bollinger_bands(df['close'], p.get('length', 20), p.get('num_std', 2.0)), 'overlay': True, 'params': {'length': 20, 'num_std': 2.0}},
    'keltner': {'fn': lambda df, **p: keltner_channels(df, p.get('length', 20), p.get('atr_length', 10), p.get('mult', 2.0)), 'overlay': True, 'params': {'length': 20, 'atr_length': 10, 'mult': 2.0}},
    'donchian': {'fn': lambda df, **p: donchian_channels(df, p.get('length', 20)), 'overlay': True, 'params': {'length': 20}},
    'vwap': {'fn': lambda df, **p: vwap(df), 'overlay': True, 'params': {}},
    'ichimoku': {'fn': lambda df, **p: ichimoku(df), 'overlay': True, 'params': {}},
    'parabolic_sar': {'fn': lambda df, **p: parabolic_sar(df, p.get('step', 0.02), p.get('max_step', 0.2)), 'overlay': True, 'params': {'step': 0.02, 'max_step': 0.2}},
    'rsi': {'fn': lambda df, **p: rsi(df['close'], p.get('length', 14)), 'overlay': False, 'params': {'length': 14}},
    'macd': {'fn': lambda df, **p: macd(df['close'], p.get('fast', 12), p.get('slow', 26), p.get('signal', 9)), 'overlay': False, 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    'stochastic': {'fn': lambda df, **p: stochastic(df, p.get('k_length', 14), p.get('k_smooth', 3), p.get('d_smooth', 3)), 'overlay': False, 'params': {'k_length': 14, 'k_smooth': 3, 'd_smooth': 3}},
    'roc': {'fn': lambda df, **p: roc(df['close'], p.get('length', 12)), 'overlay': False, 'params': {'length': 12}},
    'williams_r': {'fn': lambda df, **p: williams_r(df, p.get('length', 14)), 'overlay': False, 'params': {'length': 14}},
    'cci': {'fn': lambda df, **p: cci(df, p.get('length', 20)), 'overlay': False, 'params': {'length': 20}},
    'mfi': {'fn': lambda df, **p: mfi(df, p.get('length', 14)), 'overlay': False, 'params': {'length': 14}},
    'atr': {'fn': lambda df, **p: atr(df, p.get('length', 14)), 'overlay': False, 'params': {'length': 14}},
    'adx': {'fn': lambda df, **p: adx(df, p.get('length', 14)), 'overlay': False, 'params': {'length': 14}},
    'obv': {'fn': lambda df, **p: obv(df), 'overlay': False, 'params': {}},
    'ad_line': {'fn': lambda df, **p: ad_line(df), 'overlay': False, 'params': {}},
    'historical_volatility': {'fn': lambda df, **p: historical_volatility(df['close'], p.get('length', 20)), 'overlay': False, 'params': {'length': 20}},
}


def compute_indicator(df: pd.DataFrame, name: str, params: dict | None = None):
    """Look up an indicator by name in INDICATOR_SPECS and compute it against df."""
    spec = INDICATOR_SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unknown indicator: {name}")
    return spec['fn'](df, **(params or {}))


def compute_all(df: pd.DataFrame, names: list[str], params_by_name: dict | None = None) -> dict:
    params_by_name = params_by_name or {}
    results = {}
    for name in names:
        try:
            results[name] = compute_indicator(df, name, params_by_name.get(name))
        except Exception as exc:  # keep going even if one indicator fails on short data
            results[name] = {'error': str(exc)}
    return results
