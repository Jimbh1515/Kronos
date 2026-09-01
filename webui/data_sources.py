"""
Unified live market data access layer.

Two optional providers are supported on top of the platform's existing local
CSV/feather loading (see app.py's ``load_data_file``):

- ``yfinance``: free, no API key, covers US equities/ETFs, crypto pairs (e.g.
  ``BTC-USD``) and FX.
- ``akshare``: free, no API key, covers Chinese A-share history (matches the
  data source already used by the scripts under ``examples/``).

Both are imported lazily/optionally so the rest of the app still works if
neither package is installed (e.g. offline environments that only use local
files).
"""
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


REQUIRED_COLUMNS = ['timestamps', 'open', 'high', 'low', 'close', 'volume']


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['open', 'high', 'low', 'close']).sort_values('timestamps')
    return df.reset_index(drop=True)[REQUIRED_COLUMNS]


def fetch_yfinance(symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """period: e.g. 1mo/3mo/6mo/1y/2y/5y/10y/max. interval: 1m/5m/15m/1h/1d/1wk/1mo."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'")

    raw = raw.reset_index()
    date_col = 'Date' if 'Date' in raw.columns else 'Datetime'
    df = pd.DataFrame({
        'timestamps': pd.to_datetime(raw[date_col], utc=True).dt.tz_localize(None),
        'open': raw['Open'],
        'high': raw['High'],
        'low': raw['Low'],
        'close': raw['Close'],
        'volume': raw.get('Volume', 0),
    })
    return _finalize(df)


def fetch_akshare_a_share(symbol: str, start_date: str = '20200101', end_date: str | None = None,
                           adjust: str = 'qfq') -> pd.DataFrame:
    """symbol: 6-digit A-share code, e.g. '600580'."""
    if not AKSHARE_AVAILABLE:
        raise RuntimeError("akshare is not installed. Run: pip install akshare")

    end_date = end_date or pd.Timestamp.today().strftime('%Y%m%d')
    raw = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start_date, end_date=end_date, adjust=adjust)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for A-share symbol '{symbol}'")

    df = pd.DataFrame({
        'timestamps': pd.to_datetime(raw['日期']),
        'open': raw['开盘'],
        'high': raw['最高'],
        'low': raw['最低'],
        'close': raw['收盘'],
        'volume': raw['成交量'],
    })
    return _finalize(df)


def fetch_market_data(source: str, symbol: str, **kwargs) -> pd.DataFrame:
    if source == 'yfinance':
        return fetch_yfinance(symbol, **kwargs)
    if source == 'akshare':
        return fetch_akshare_a_share(symbol, **kwargs)
    raise ValueError(f"Unknown data source: {source}")


def search_symbols(query: str, limit: int = 10) -> list:
    """Best-effort ticker/name search, used by the screener and TA tabs' symbol pickers."""
    if not YFINANCE_AVAILABLE or not query:
        return []
    try:
        quotes = yf.Search(query, max_results=limit).quotes
        return [{
            'symbol': q.get('symbol'),
            'name': q.get('shortname') or q.get('longname') or '',
            'type': q.get('quoteType'),
            'exchange': q.get('exchange'),
        } for q in quotes if q.get('symbol')]
    except Exception:
        return []


def fetch_options_chain(symbol: str, expiry: str | None = None) -> dict:
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    ticker = yf.Ticker(symbol)
    expirations = list(ticker.options)
    if not expirations:
        raise ValueError(f"No listed options found for '{symbol}'")

    chosen = expiry if expiry in expirations else expirations[0]
    chain = ticker.option_chain(chosen)
    hist = ticker.history(period='1d')
    spot = float(hist['Close'].iloc[-1]) if not hist.empty else None

    def _records(frame):
        cols = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume',
                 'openInterest', 'impliedVolatility', 'inTheMoney']
        available = [c for c in cols if c in frame.columns]
        return frame[available].to_dict('records')

    return {
        'symbol': symbol,
        'spot': spot,
        'expiry': chosen,
        'expirations': expirations,
        'calls': _records(chain.calls),
        'puts': _records(chain.puts),
    }


def data_sources_status() -> dict:
    return {'yfinance': YFINANCE_AVAILABLE, 'akshare': AKSHARE_AVAILABLE}
