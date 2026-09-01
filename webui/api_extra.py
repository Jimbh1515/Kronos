"""
Flask routes for the financial-analyst platform's added modules: live market
data, technical indicators, risk analytics, backtesting, options analytics,
and the multi-symbol screener/watchlist. Kept in its own file and wired into
app.py via ``register_routes`` so the original Kronos prediction routes are
left untouched.
"""
import numpy as np
import pandas as pd
from flask import request, jsonify

import data_sources
from analytics import indicators as ta
from analytics import risk as risk_mod
from analytics import backtest as backtest_mod
from analytics import options as options_mod
from analytics import screener as screener_mod


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------

def _clean(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if (np.isnan(value) or np.isinf(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _timestamps_list(df: pd.DataFrame):
    if 'timestamps' in df.columns:
        return [ts.isoformat() for ts in pd.to_datetime(df['timestamps'])]
    return list(range(len(df)))


def _series_payload(series: pd.Series, timestamps: list):
    return [{'t': t, 'v': _clean(v)} for t, v in zip(timestamps, series.tolist())]


def _indicator_payload(result, timestamps: list):
    if isinstance(result, dict) and 'error' in result:
        return result
    if isinstance(result, pd.DataFrame):
        return {'type': 'multi', 'series': {col: _series_payload(result[col], timestamps) for col in result.columns}}
    return {'type': 'series', 'series': _series_payload(result, timestamps)}


def _ohlcv_records(df: pd.DataFrame):
    out = []
    timestamps = _timestamps_list(df)
    for ts, (_, row) in zip(timestamps, df.iterrows()):
        out.append({
            'timestamp': ts,
            'open': _clean(row.get('open')),
            'high': _clean(row.get('high')),
            'low': _clean(row.get('low')),
            'close': _clean(row.get('close')),
            'volume': _clean(row.get('volume')) if 'volume' in df.columns else None,
        })
    return out


# ---------------------------------------------------------------------------
# Data resolution shared by every analytics endpoint: either a local file
# (matching the existing /api/load-data flow) or a live source+symbol pull.
# ---------------------------------------------------------------------------

def _resolve_df(payload: dict, load_data_file) -> pd.DataFrame:
    file_path = payload.get('file_path')
    if file_path:
        df, error = load_data_file(file_path)
        if error:
            raise ValueError(error)
        return df

    source = payload.get('source')
    symbol = payload.get('symbol')
    if source and symbol:
        kwargs = {}
        if payload.get('period'):
            kwargs['period'] = payload['period']
        if payload.get('interval'):
            kwargs['interval'] = payload['interval']
        if payload.get('start_date'):
            kwargs['start_date'] = payload['start_date']
        if payload.get('end_date'):
            kwargs['end_date'] = payload['end_date']
        return data_sources.fetch_market_data(source, symbol, **kwargs)

    raise ValueError('Provide either file_path, or source + symbol')


def register_routes(app, load_data_file):

    # ------------------------------------------------------------------ #
    # Live market data
    # ------------------------------------------------------------------ #

    @app.route('/api/data-sources-status')
    def data_sources_status():
        return jsonify(data_sources.data_sources_status())

    @app.route('/api/symbol-search', methods=['POST'])
    def symbol_search():
        query = (request.get_json() or {}).get('query', '')
        return jsonify({'results': data_sources.search_symbols(query)})

    @app.route('/api/market-data', methods=['POST'])
    def market_data():
        try:
            payload = request.get_json() or {}
            df = _resolve_df(payload, load_data_file)
            return jsonify({
                'success': True,
                'rows': len(df),
                'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else None,
                'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else None,
                'candles': _ohlcv_records(df),
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    # ------------------------------------------------------------------ #
    # Technical indicators
    # ------------------------------------------------------------------ #

    @app.route('/api/indicators/list')
    def indicators_list():
        return jsonify({
            name: {'overlay': spec['overlay'], 'default_params': spec['params']}
            for name, spec in ta.INDICATOR_SPECS.items()
        })

    @app.route('/api/indicators', methods=['POST'])
    def indicators_compute():
        try:
            payload = request.get_json() or {}
            df = _resolve_df(payload, load_data_file)
            names = payload.get('indicators') or []
            if not names:
                return jsonify({'error': 'Provide at least one indicator name'}), 400
            params_by_name = payload.get('params') or {}

            timestamps = _timestamps_list(df)
            results = ta.compute_all(df, names, params_by_name)
            return jsonify({
                'success': True,
                'candles': _ohlcv_records(df),
                'indicators': {name: _indicator_payload(res, timestamps) for name, res in results.items()},
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    # ------------------------------------------------------------------ #
    # Risk analytics
    # ------------------------------------------------------------------ #

    @app.route('/api/risk', methods=['POST'])
    def risk_analysis():
        try:
            payload = request.get_json() or {}
            df = _resolve_df(payload, load_data_file)
            risk_free_rate = float(payload.get('risk_free_rate', 0.0))
            periods_per_year = int(payload.get('periods_per_year', 252))

            benchmark_prices = None
            benchmark_payload = payload.get('benchmark')
            if benchmark_payload:
                bench_df = _resolve_df(benchmark_payload, load_data_file)
                benchmark_prices = bench_df['close']

            summary = risk_mod.risk_summary(df['close'], risk_free_rate, periods_per_year, benchmark_prices)
            timestamps = _timestamps_list(df)
            returns = risk_mod.simple_returns(df['close'])
            rolling_vol = risk_mod.rolling_volatility(returns, min(21, max(2, len(returns) // 4 or 1)), periods_per_year)
            drawdown = risk_mod.max_drawdown(df['close'])['drawdown_series']

            return jsonify({
                'success': True,
                'summary': _clean(summary),
                'drawdown_series': _series_payload(drawdown, timestamps),
                'rolling_volatility_series': _series_payload(rolling_vol, timestamps),
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/risk/correlation', methods=['POST'])
    def risk_correlation():
        try:
            payload = request.get_json() or {}
            items = payload.get('symbols') or []  # [{file_path|source+symbol, label}]
            price_dict = {}
            for item in items:
                df = _resolve_df(item, load_data_file)
                label = item.get('label') or item.get('symbol') or item.get('file_path')
                price_dict[label] = df.set_index('timestamps')['close'] if 'timestamps' in df.columns else df['close']
            corr = risk_mod.correlation_matrix(price_dict)
            corr = corr.fillna(0)
            return jsonify({'success': True, 'labels': list(corr.columns), 'matrix': corr.values.tolist()})
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    # ------------------------------------------------------------------ #
    # Backtesting
    # ------------------------------------------------------------------ #

    @app.route('/api/backtest/strategies')
    def backtest_strategies():
        return jsonify({
            name: {'label': spec['label'], 'description': spec['description'], 'default_params': spec['params']}
            for name, spec in backtest_mod.STRATEGY_REGISTRY.items()
        })

    @app.route('/api/backtest', methods=['POST'])
    def backtest_run():
        try:
            payload = request.get_json() or {}
            df = _resolve_df(payload, load_data_file)
            strategy = payload.get('strategy')
            if not strategy:
                return jsonify({'error': 'Provide a strategy name'}), 400

            result = backtest_mod.run_backtest(
                df,
                strategy,
                params=payload.get('params'),
                initial_capital=float(payload.get('initial_capital', 100_000.0)),
                fee_bps=float(payload.get('fee_bps', 5.0)),
                allow_short=bool(payload.get('allow_short', True)),
            )
            timestamps = _timestamps_list(df)
            return jsonify({
                'success': True,
                'stats': _clean(result['stats']),
                'trades': _clean(result['trades']),
                'timestamps': timestamps,
                'equity_curve': _clean(result['equity_curve']),
                'buy_hold_curve': _clean(result['buy_hold_curve']),
                'positions': _clean(result['positions']),
                'candles': _ohlcv_records(df),
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    # ------------------------------------------------------------------ #
    # Options analytics
    # ------------------------------------------------------------------ #

    @app.route('/api/options/analyze', methods=['POST'])
    def options_analyze():
        try:
            p = request.get_json() or {}
            result = options_mod.full_option_analysis(
                spot=float(p['spot']), strike=float(p['strike']), days_to_expiry=float(p['days_to_expiry']),
                r=float(p.get('r', 0.05)), sigma=float(p['sigma']), option_type=p.get('option_type', 'call'),
                q=float(p.get('q', 0.0)),
            )
            return jsonify({'success': True, 'result': _clean(result)})
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/options/implied-vol', methods=['POST'])
    def options_implied_vol():
        try:
            p = request.get_json() or {}
            iv = options_mod.implied_volatility(
                market_price=float(p['market_price']), spot=float(p['spot']), strike=float(p['strike']),
                t=float(p['days_to_expiry']) / 365, r=float(p.get('r', 0.05)),
                option_type=p.get('option_type', 'call'), q=float(p.get('q', 0.0)),
            )
            return jsonify({'success': True, 'implied_volatility': iv})
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/options/payoff', methods=['POST'])
    def options_payoff():
        try:
            p = request.get_json() or {}
            spot = float(p['spot'])
            if p.get('legs'):
                legs = p['legs']
            else:
                builder = options_mod.STRATEGY_LEGS.get(p.get('strategy', ''))
                if not builder:
                    return jsonify({'error': f"Unknown strategy: {p.get('strategy')}"}), 400
                legs = builder(float(p['strike']), float(p['premium']))
            result = options_mod.strategy_payoff(spot, legs, spread_pct=float(p.get('spread_pct', 0.3)))
            return jsonify({'success': True, 'result': _clean(result)})
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/options/strategies')
    def options_strategies():
        return jsonify({'strategies': list(options_mod.STRATEGY_LEGS.keys())})

    @app.route('/api/options/chain', methods=['POST'])
    def options_chain():
        try:
            p = request.get_json() or {}
            result = data_sources.fetch_options_chain(p['symbol'], p.get('expiry'))
            return jsonify({'success': True, 'result': _clean(result)})
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    # ------------------------------------------------------------------ #
    # Screener & watchlists
    # ------------------------------------------------------------------ #

    @app.route('/api/screener/scan', methods=['POST'])
    def screener_scan():
        try:
            p = request.get_json() or {}
            symbols = p.get('symbols') or []
            if not symbols:
                return jsonify({'error': 'Provide at least one symbol'}), 400
            source = p.get('source', 'yfinance')
            period = p.get('period', '6mo')
            interval = p.get('interval', '1d')

            def fetch_fn(sym):
                return data_sources.fetch_market_data(source, sym, period=period, interval=interval)

            results = screener_mod.scan_symbols(symbols, fetch_fn)
            return jsonify({'success': True, 'results': _clean(results)})
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/watchlists', methods=['GET'])
    def watchlists_list():
        return jsonify(screener_mod.list_watchlists())

    @app.route('/api/watchlists', methods=['POST'])
    def watchlists_save():
        p = request.get_json() or {}
        name = p.get('name')
        if not name:
            return jsonify({'error': 'Provide a watchlist name'}), 400
        store = screener_mod.save_watchlist(name, p.get('symbols', []))
        return jsonify(store)

    @app.route('/api/watchlists/<name>', methods=['DELETE'])
    def watchlists_delete(name):
        return jsonify(screener_mod.delete_watchlist(name))

    @app.route('/api/watchlists/<name>/symbols', methods=['POST'])
    def watchlists_add_symbol(name):
        p = request.get_json() or {}
        symbol = p.get('symbol')
        if not symbol:
            return jsonify({'error': 'Provide a symbol'}), 400
        return jsonify(screener_mod.add_symbol(name, symbol))

    @app.route('/api/watchlists/<name>/symbols/<symbol>', methods=['DELETE'])
    def watchlists_remove_symbol(name, symbol):
        return jsonify(screener_mod.remove_symbol(name, symbol))
