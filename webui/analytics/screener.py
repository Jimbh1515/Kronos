"""
Multi-symbol screener: fetches each symbol's recent price history through a
caller-supplied fetch function (so it works against yfinance, akshare, or
local files interchangeably), computes a standard set of technical signals,
and returns a sortable/filterable table plus JSON-backed watchlist storage.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import indicators as ta

WATCHLIST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'storage', 'watchlists.json')


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame) -> dict:
    """One row of screener signals for a single symbol's OHLCV history."""
    if df is None or len(df) < 5:
        return {'error': 'insufficient data'}

    close = df['close']
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last

    sma20 = ta.sma(close, 20)
    sma50 = ta.sma(close, 50)
    sma200 = ta.sma(close, 200)
    r = ta.rsi(close, 14)
    m = ta.macd(close)
    bb = ta.bollinger_bands(close, 20, 2.0)
    atr_val = ta.atr(df, 14)
    roc20 = ta.roc(close, 20)

    high_52w = float(close.tail(252).max()) if len(close) >= 5 else last
    low_52w = float(close.tail(252).min()) if len(close) >= 5 else last

    def _tail(series):
        val = series.iloc[-1]
        return None if pd.isna(val) else float(val)

    trend = 'neutral'
    s50, s200 = _tail(sma50), _tail(sma200)
    if s50 is not None and s200 is not None:
        trend = 'bullish' if s50 > s200 else 'bearish'

    macd_state = 'neutral'
    macd_last, signal_last = _tail(m['macd']), _tail(m['signal'])
    if macd_last is not None and signal_last is not None:
        macd_state = 'bullish' if macd_last > signal_last else 'bearish'

    rsi_last = _tail(r)
    rsi_state = 'neutral'
    if rsi_last is not None:
        if rsi_last >= 70:
            rsi_state = 'overbought'
        elif rsi_last <= 30:
            rsi_state = 'oversold'

    # Simple composite score in [-3, 3]: trend + macd + rsi mean-reversion tilt.
    score = 0
    score += 1 if trend == 'bullish' else (-1 if trend == 'bearish' else 0)
    score += 1 if macd_state == 'bullish' else (-1 if macd_state == 'bearish' else 0)
    score += 1 if rsi_state == 'oversold' else (-1 if rsi_state == 'overbought' else 0)

    return {
        'last_price': last,
        'change_pct': (last / prev - 1) * 100 if prev else 0.0,
        'sma20': _tail(sma20),
        'sma50': s50,
        'sma200': s200,
        'rsi_14': rsi_last,
        'rsi_state': rsi_state,
        'macd_state': macd_state,
        'trend': trend,
        'bollinger_percent_b': _tail(bb['percent_b']),
        'atr_14': _tail(atr_val),
        'roc_20_pct': _tail(roc20),
        'pct_from_52w_high': (last / high_52w - 1) * 100 if high_52w else 0.0,
        'pct_from_52w_low': (last / low_52w - 1) * 100 if low_52w else 0.0,
        'volume': float(df['volume'].iloc[-1]) if 'volume' in df.columns and len(df) else None,
        'score': score,
    }


def scan_symbols(symbols: list, fetch_fn, max_workers: int = 8) -> list:
    """
    fetch_fn(symbol) -> DataFrame with OHLCV; called concurrently since it's
    typically a network request (yfinance/akshare) per symbol.
    """
    results = []

    def _run(sym):
        try:
            df = fetch_fn(sym)
            signals = compute_signals(df)
            return {'symbol': sym, **signals}
        except Exception as exc:
            return {'symbol': sym, 'error': str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, sym): sym for sym in symbols}
        for future in as_completed(futures):
            results.append(future.result())

    order = {sym: i for i, sym in enumerate(symbols)}
    results.sort(key=lambda r: order.get(r['symbol'], 0))
    return results


# ---------------------------------------------------------------------------
# Watchlist persistence (JSON file, mirrors the repo's existing pattern of
# writing prediction results to disk instead of a database).
# ---------------------------------------------------------------------------

def _load_store() -> dict:
    if not os.path.exists(WATCHLIST_PATH):
        return {}
    try:
        with open(WATCHLIST_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def list_watchlists() -> dict:
    return _load_store()


def save_watchlist(name: str, symbols: list) -> dict:
    store = _load_store()
    store[name] = sorted(set(s.strip().upper() for s in symbols if s.strip()))
    _save_store(store)
    return store


def delete_watchlist(name: str) -> dict:
    store = _load_store()
    store.pop(name, None)
    _save_store(store)
    return store


def add_symbol(name: str, symbol: str) -> dict:
    store = _load_store()
    symbols = set(store.get(name, []))
    symbols.add(symbol.strip().upper())
    store[name] = sorted(symbols)
    _save_store(store)
    return store


def remove_symbol(name: str, symbol: str) -> dict:
    store = _load_store()
    symbols = set(store.get(name, []))
    symbols.discard(symbol.strip().upper())
    store[name] = sorted(symbols)
    _save_store(store)
    return store
