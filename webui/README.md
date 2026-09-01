# Kronos Financial Analyst Platform

Web user interface built around the Kronos financial prediction model, extended into a full
technical-analysis / risk / backtesting / options / screening platform — everything a discretionary
or systematic analyst needs in one app, on top of Kronos's own K-line forecasting.

## ✨ Features

### 🔮 Prediction (original Kronos functionality — unchanged)
- **Multi-format data support**: Supports CSV, Feather and other financial data formats
- **Smart time window**: Fixed 400+120 data point time window slider selection
- **Real model prediction**: Integrated real Kronos model, supports multiple model sizes
- **Prediction quality control**: Adjustable temperature, nucleus sampling, sample count and other parameters
- **Multi-device support**: Supports CPU, CUDA, MPS and other computing devices
- **Comparison analysis**: Detailed comparison between prediction results and actual data
- **K-line chart display**: Professional financial K-line chart display

### 📊 Technical Analysis
Interactive candlestick charting with 20 indicators computed from a pure pandas/numpy library
(`webui/analytics/indicators.py`, no compiled TA-Lib dependency): SMA, EMA, WMA, Bollinger Bands,
Keltner Channels, Donchian Channels, VWAP, Ichimoku Cloud, Parabolic SAR, RSI, MACD, Stochastic,
ROC, Williams %R, CCI, MFI, ATR, ADX/DI, OBV, Accumulation/Distribution, and historical volatility —
plus classic floor-trader pivot points and Fibonacci retracement levels. Overlays render on the price
panel; oscillators get their own stacked, x-axis-synced panel below.

### 🔍 Screener & Watchlists
Scan any list of symbols concurrently (via yfinance or akshare) for trend/RSI/MACD signals and a
composite score; save/load/edit named watchlists (persisted to `webui/storage/watchlists.json`).

### 💰 Backtest
Vectorized, look-ahead-bias-free backtesting engine (`webui/analytics/backtest.py`) with six built-in
rule-based strategies — SMA crossover, EMA crossover, RSI mean reversion, MACD crossover, Bollinger
Band reversion, Donchian breakout — configurable parameters, transaction costs, optional shorting, an
equity curve vs. buy-and-hold, a full trade log, and performance stats (CAGR, Sharpe/Sortino/Calmar,
max drawdown, win rate, profit factor, exposure).

### ⚠️ Risk Analytics
Single-asset risk reports (annualized return/volatility, Sharpe, Sortino, Calmar, max drawdown with
peak/trough/recovery dates, historical & parametric VaR, CVaR/Expected Shortfall, skewness, kurtosis,
beta vs. a benchmark) plus a drawdown chart, rolling volatility chart, and a multi-symbol return
correlation heatmap.

### 🎯 Options
Black-Scholes pricing with continuous dividend yield, all five Greeks (delta, gamma, vega, theta,
rho), a Newton-Raphson (with bisection fallback) implied-volatility solver, and payoff/breakeven
diagrams for common strategies (long call/put, covered call, protective put, straddle) plus live
option-chain lookups where the underlying is available on Yahoo Finance.

### 🌐 Live Market Data
The Technical Analysis, Screener, Backtest, and Risk tabs all accept either a local file (the
original workflow) or a live symbol — via `yfinance` (US/global equities, ETFs, crypto pairs like
`BTC-USD`, FX) or `akshare` (Chinese A-shares, matching the data source already used by the scripts
under `examples/`).

## 🚀 Quick Start

### Method 1: Start with Python script
```bash
cd webui
python run.py
```

### Method 2: Start with Shell script
```bash
cd webui
chmod +x start.sh
./start.sh
```

### Method 3: Start Flask application directly
```bash
cd webui
python app.py
```

After successful startup, visit http://localhost:7070

### ☁️ One-Click Deploy to Render

The repo includes a Render Blueprint (`render.yaml`) that builds `webui/` as a web service with
a CPU-only PyTorch wheel (much smaller/faster than the default GPU-bundled wheel) and serves it
with `gunicorn`.

1. Click **[Deploy to Render](https://render.com/deploy?repo=https://github.com/Jimbh1515/Kronos)**.
2. When Render's setup wizard asks for a branch, pick `claude/financial-analyst-platform-7pjlov`
   (or `master` once the PR is merged).
3. Click **Apply** — Render will build and give you a public `https://<your-service>.onrender.com` URL.

**Notes:**
- The free instance type has 512MB RAM. `torch` + Flask + pandas + scipy already use a meaningful
  chunk of that at idle, so loading a Kronos model (the Prediction tab) may be tight — **Kronos-mini**
  (4.1M params) is the most likely to fit; Kronos-small/base may need a paid instance type with more RAM.
- The Technical Analysis / Backtest / Risk / Options / Screener tabs don't load the model at all and
  should run fine on the free tier.
- The free tier spins down after 15 minutes idle and cold-starts on the next request (10–30s delay).
- If the Blueprint's fields have drifted from Render's current schema, create the service manually
  instead: **New + → Web Service → connect this repo**, root directory `webui`, build command
  `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`,
  start command `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`.

## 📋 Usage Steps

1. **Load data**: Select financial data file from data directory
2. **Load model**: Select Kronos model and computing device
3. **Set parameters**: Adjust prediction quality parameters
4. **Select time window**: Use slider to select 400+120 data point time range
5. **Start prediction**: Click prediction button to generate results
6. **View results**: View prediction results in charts and tables

## 🔧 Prediction Quality Parameters

### Temperature (T)
- **Range**: 0.1 - 2.0
- **Effect**: Controls prediction randomness
- **Recommendation**: 1.2-1.5 for better prediction quality

### Nucleus Sampling (top_p)
- **Range**: 0.1 - 1.0
- **Effect**: Controls prediction diversity
- **Recommendation**: 0.95-1.0 to consider more possibilities

### Sample Count
- **Range**: 1 - 5
- **Effect**: Generate multiple prediction samples
- **Recommendation**: 2-3 samples to improve quality

## 📊 Supported Data Formats

### Required Columns
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price

### Optional Columns
- `volume`: Trading volume
- `amount`: Trading amount (not used for prediction)
- `timestamps`/`timestamp`/`date`: Timestamp

## 🤖 Model Support

- **Kronos-mini**: 4.1M parameters, lightweight fast prediction
- **Kronos-small**: 24.7M parameters, balanced performance and speed
- **Kronos-base**: 102.3M parameters, high quality prediction

## 🖥️ GPU Acceleration Support

- **CPU**: General computing, best compatibility
- **CUDA**: NVIDIA GPU acceleration, best performance
- **MPS**: Apple Silicon GPU acceleration, recommended for Mac users

## ⚠️ Notes

- `amount` column is not used for prediction, only for display
- Time window is fixed at 400+120=520 data points
- Ensure data file contains sufficient historical data
- First model loading may require download, please be patient

## 🔍 Comparison Analysis

The system automatically provides comparison analysis between prediction results and actual data, including:
- Price difference statistics
- Error analysis
- Prediction quality assessment

## 🌐 REST API Reference (new in this platform)

All endpoints accept/return JSON. Every analytics endpoint (`/api/indicators`, `/api/risk`,
`/api/backtest`, `/api/screener/scan`) resolves its price data the same way: pass either
`{"file_path": "..."}` for a local file, or `{"source": "yfinance"|"akshare", "symbol": "...",
"period": "6mo", "interval": "1d"}` for a live pull.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/data-sources-status` | GET | Which of yfinance/akshare are installed |
| `/api/symbol-search` | POST | Ticker/name search (yfinance-backed) |
| `/api/market-data` | POST | Fetch raw OHLCV candles |
| `/api/indicators/list` | GET | Available indicators + default params |
| `/api/indicators` | POST | Compute one or more indicators over a symbol/file |
| `/api/risk` | POST | Full risk summary + drawdown/rolling-vol series |
| `/api/risk/correlation` | POST | Return-correlation matrix across symbols |
| `/api/backtest/strategies` | GET | Available strategies + default params |
| `/api/backtest` | POST | Run a backtest, get stats/equity curve/trade log |
| `/api/options/analyze` | POST | Black-Scholes price + Greeks |
| `/api/options/implied-vol` | POST | Solve implied volatility from a market price |
| `/api/options/payoff` | POST | Strategy payoff diagram at expiry |
| `/api/options/chain` | POST | Live option chain (yfinance) |
| `/api/screener/scan` | POST | Scan a symbol list for signals |
| `/api/watchlists` | GET/POST | List / create-or-update watchlists |
| `/api/watchlists/<name>` | DELETE | Delete a watchlist |
| `/api/watchlists/<name>/symbols` | POST | Add a symbol to a watchlist |
| `/api/watchlists/<name>/symbols/<symbol>` | DELETE | Remove a symbol from a watchlist |

## 🛠️ Technical Architecture

- **Backend**: Flask + Python (`app.py` for the original Kronos prediction routes,
  `api_extra.py` for the new analytics/data/screener/options routes)
- **Analytics**: `analytics/indicators.py`, `analytics/risk.py`, `analytics/backtest.py`,
  `analytics/options.py`, `analytics/screener.py` — pure pandas/numpy/scipy, no compiled
  TA-Lib dependency
- **Live data**: `data_sources.py` (yfinance + optional akshare)
- **Frontend**: HTML + CSS + JavaScript, tabbed single-page layout
- **Charts**: Plotly.js
- **Data processing**: Pandas + NumPy + SciPy
- **Model**: Hugging Face Transformers

## 📝 Troubleshooting

### Common Issues
1. **Port occupied**: Modify port number in app.py
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model loading failed**: Check network connection and model ID
4. **Data format error**: Ensure data column names and format are correct

### Log Viewing
Detailed runtime information will be displayed in the console at startup, including model status and error messages.

## 📄 License

This project follows the license terms of the original Kronos project.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this Web UI!

## 📞 Support

If you have questions, please check:
1. Project documentation
2. GitHub Issues
3. Console error messages
