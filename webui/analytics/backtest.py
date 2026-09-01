"""
Vectorized single-asset backtesting engine.

A "strategy" is a function ``(df) -> pd.Series`` returning a position in
{-1, 0, 1} (short/flat/long) for every bar. Positions are applied to the
*next* bar's return to avoid look-ahead bias. Built-in strategies cover the
most common rule-based approaches; custom ones can be registered by name.
"""
import numpy as np
import pandas as pd

from . import indicators as ta
from . import risk as risk_mod


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

def strategy_sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = ta.sma(df['close'], fast)
    slow_ma = ta.sma(df['close'], slow)
    return pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=df.index).where(~(fast_ma.isna() | slow_ma.isna()), 0)


def strategy_ema_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    fast_ma = ta.ema(df['close'], fast)
    slow_ma = ta.ema(df['close'], slow)
    return pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=df.index).where(~(fast_ma.isna() | slow_ma.isna()), 0)


def strategy_rsi_mean_reversion(df: pd.DataFrame, length: int = 14, lower: float = 30, upper: float = 70) -> pd.Series:
    r = ta.rsi(df['close'], length)
    pos = pd.Series(0, index=df.index, dtype=float)
    pos = pos.mask(r < lower, 1).mask(r > upper, -1)
    return pos.replace(0, np.nan).ffill().fillna(0)


def strategy_macd_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ta.macd(df['close'], fast, slow, signal)
    return pd.Series(np.where(m['macd'] > m['signal'], 1, -1), index=df.index).where(~(m['macd'].isna() | m['signal'].isna()), 0)


def strategy_bollinger_reversion(df: pd.DataFrame, length: int = 20, num_std: float = 2.0) -> pd.Series:
    bb = ta.bollinger_bands(df['close'], length, num_std)
    pos = pd.Series(0, index=df.index, dtype=float)
    pos = pos.mask(df['close'] < bb['lower'], 1).mask(df['close'] > bb['upper'], -1)
    pos = pos.mask((df['close'] >= bb['lower']) & (df['close'] <= bb['mid']) & (pos.shift(1) == 1), 1)
    return pos.replace(0, np.nan).ffill().fillna(0).where(~bb['mid'].isna(), 0)


def strategy_donchian_breakout(df: pd.DataFrame, length: int = 20) -> pd.Series:
    ch = ta.donchian_channels(df, length)
    pos = pd.Series(np.nan, index=df.index)
    pos = pos.mask(df['close'] >= ch['upper'], 1).mask(df['close'] <= ch['lower'], -1)
    return pos.ffill().fillna(0)


STRATEGY_REGISTRY = {
    'sma_crossover': {'fn': strategy_sma_crossover, 'params': {'fast': 20, 'slow': 50},
                       'label': 'SMA Crossover', 'description': 'Long when fast SMA > slow SMA, short otherwise.'},
    'ema_crossover': {'fn': strategy_ema_crossover, 'params': {'fast': 12, 'slow': 26},
                       'label': 'EMA Crossover', 'description': 'Long when fast EMA > slow EMA, short otherwise.'},
    'rsi_mean_reversion': {'fn': strategy_rsi_mean_reversion, 'params': {'length': 14, 'lower': 30, 'upper': 70},
                            'label': 'RSI Mean Reversion', 'description': 'Long when RSI < lower band, short when RSI > upper band, hold otherwise.'},
    'macd_crossover': {'fn': strategy_macd_crossover, 'params': {'fast': 12, 'slow': 26, 'signal': 9},
                        'label': 'MACD Crossover', 'description': 'Long when MACD line > signal line, short otherwise.'},
    'bollinger_reversion': {'fn': strategy_bollinger_reversion, 'params': {'length': 20, 'num_std': 2.0},
                             'label': 'Bollinger Band Reversion', 'description': 'Buy at the lower band, exit at the midline; short at the upper band.'},
    'donchian_breakout': {'fn': strategy_donchian_breakout, 'params': {'length': 20},
                           'label': 'Donchian Channel Breakout', 'description': 'Long on an upper-channel breakout, short on a lower-channel breakdown.'},
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, strategy_name: str, params: dict | None = None,
                  initial_capital: float = 100_000.0, fee_bps: float = 5.0,
                  allow_short: bool = True, periods_per_year: int = 252) -> dict:
    """
    Run a vectorized backtest of a registered strategy over df (needs open/high/low/close,
    ideally volume). fee_bps is a round-trip-agnostic per-trade cost applied to position changes.
    """
    spec = STRATEGY_REGISTRY.get(strategy_name)
    if spec is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    df = df.reset_index(drop=True).copy()
    positions = spec['fn'](df, **(params or spec['params']))
    if not allow_short:
        positions = positions.clip(lower=0)

    # Trade on next bar's return to avoid look-ahead bias.
    market_returns = df['close'].pct_change().fillna(0)
    applied_positions = positions.shift(1).fillna(0)

    trade_events = applied_positions.diff().fillna(applied_positions.iloc[0] if len(applied_positions) else 0)
    fee_cost = trade_events.abs() * (fee_bps / 10_000)

    strategy_returns = applied_positions * market_returns - fee_cost
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    buy_hold_curve = initial_capital * (1 + market_returns).cumprod()

    # Build trade log from position changes.
    trades = []
    entry_idx = None
    entry_pos = 0
    for i in range(len(applied_positions)):
        cur = applied_positions.iloc[i]
        if cur != entry_pos:
            if entry_pos != 0 and entry_idx is not None:
                entry_price = df['close'].iloc[entry_idx]
                exit_price = df['close'].iloc[i]
                pnl_pct = (exit_price / entry_price - 1) * entry_pos * 100
                trades.append({
                    'direction': 'long' if entry_pos > 0 else 'short',
                    'entry_index': int(entry_idx),
                    'exit_index': int(i),
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'pnl_pct': float(pnl_pct),
                })
            entry_idx = i if cur != 0 else None
            entry_pos = cur

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    gross_profit = sum(t['pnl_pct'] for t in wins)
    gross_loss = abs(sum(t['pnl_pct'] for t in losses))

    strategy_returns_clean = strategy_returns.dropna()
    risk_stats = risk_mod.risk_summary(equity_curve, periods_per_year=periods_per_year)

    stats = {
        'strategy': strategy_name,
        'params': params or spec['params'],
        'initial_capital': initial_capital,
        'final_equity': float(equity_curve.iloc[-1]) if len(equity_curve) else initial_capital,
        'total_return_pct': risk_stats['total_return_pct'],
        'buy_hold_return_pct': float((buy_hold_curve.iloc[-1] / initial_capital - 1) * 100) if len(buy_hold_curve) else 0.0,
        'annualized_return_pct': risk_stats['annualized_return_pct'],
        'annualized_volatility_pct': risk_stats['annualized_volatility_pct'],
        'sharpe_ratio': risk_stats['sharpe_ratio'],
        'sortino_ratio': risk_stats['sortino_ratio'],
        'max_drawdown_pct': risk_stats['max_drawdown_pct'],
        'calmar_ratio': risk_stats['calmar_ratio'],
        'num_trades': len(trades),
        'win_rate_pct': (len(wins) / len(trades) * 100) if trades else 0.0,
        'avg_win_pct': (gross_profit / len(wins)) if wins else 0.0,
        'avg_loss_pct': (-gross_loss / len(losses)) if losses else 0.0,
        'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0),
        'exposure_pct': float((applied_positions != 0).mean() * 100),
    }
    if stats['profit_factor'] == float('inf'):
        stats['profit_factor'] = None  # JSON-safe

    return {
        'stats': stats,
        'trades': trades[-200:],  # cap payload size
        'equity_curve': equity_curve.tolist(),
        'buy_hold_curve': buy_hold_curve.tolist(),
        'positions': positions.tolist(),
    }
