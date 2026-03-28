"""
Phase 19 Level 3: Free Market Data Fetcher

Fetches derivatives, DeFi, and macro data from free APIs:
1. Bybit derivatives: Open Interest, Funding Rate, Long/Short Ratio (no key)
2. DeFi Llama: Total TVL, Stablecoin supply changes (no key)
3. FRED: DXY proxy (Trade-Weighted USD), Treasury yields (free key)

All data stored in SQLite for RAG pipeline consumption.
Scheduler runs every 15 min for derivatives, hourly for DeFi/macro.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(__file__))

from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)

# HTTP client — reuse across calls
_http_client = None


def _get_http():
    """Lazy singleton httpx client with timeouts."""
    global _http_client
    if _http_client is None:
        import httpx
        _http_client = httpx.Client(timeout=15.0, follow_redirects=True)
    return _http_client


class MarketDataFetcher:
    """
    Fetches market data from free public APIs.
    Each fetch method is independent and gracefully degrades on failure.
    """

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self):
        """Create tables for market data storage."""
        try:
            with self._get_conn() as conn:
                # Derivatives data (Bybit)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS derivatives_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        open_interest_usd REAL,
                        funding_rate REAL,
                        long_short_ratio REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_pair_ts ON derivatives_data(pair, timestamp)")

                # Macro data (FRED, yfinance, etc.)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS macro_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        prev_value REAL,
                        change_pct REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_name_ts ON macro_data(metric_name, timestamp)")

                # DeFi data (DeFi Llama)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS defi_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        change_pct REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_defi_name_ts ON defi_data(metric_name, timestamp)")

                # Google Trends search interest
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_trends (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword TEXT NOT NULL,
                        interest_score INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trends_kw_ts ON search_trends(keyword, timestamp)")

                conn.commit()
        except Exception as e:
            logger.error(f"[MarketDataFetcher] Table init failed: {e}")

    # ── Bybit Derivatives ──────────────────────────────────────

    def fetch_derivatives(self, pairs: List[str] = None):
        """
        Fetch OI, funding rate, long/short ratio from Bybit public API.
        No API key required.
        """
        if pairs is None:
            pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        http = _get_http()
        fetched = 0

        for symbol in pairs:
            try:
                oi_val = self._fetch_bybit_oi(http, symbol)
                fr_val = self._fetch_bybit_funding(http, symbol)
                ls_val = self._fetch_bybit_ls_ratio(http, symbol)

                # Store
                pair_formatted = symbol.replace("USDT", "/USDT")
                with self._get_conn() as conn:
                    conn.execute("""
                        INSERT INTO derivatives_data (pair, open_interest_usd, funding_rate, long_short_ratio)
                        VALUES (?, ?, ?, ?)
                    """, (pair_formatted, oi_val, fr_val, ls_val))
                    conn.commit()

                fetched += 1
                logger.info(f"[MarketDataFetcher:Deriv] {pair_formatted}: OI=${oi_val:,.0f}, FR={fr_val:.6f}, L/S={ls_val:.3f}" if oi_val else f"[MarketDataFetcher:Deriv] {pair_formatted}: partial data")

            except Exception as e:
                logger.warning(f"[MarketDataFetcher:Deriv] {symbol} failed: {e}")

        return fetched

    def _fetch_bybit_oi(self, http, symbol: str) -> Optional[float]:
        """Fetch open interest from Bybit v5 API."""
        try:
            resp = http.get(
                "https://api.bybit.com/v5/market/open-interest",
                params={"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": 1}
            )
            data = resp.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                return float(data["result"]["list"][0].get("openInterest", 0))
        except Exception as e:
            logger.debug(f"[MarketDataFetcher] Bybit OI failed for {symbol}: {e}")
        return None

    def _fetch_bybit_funding(self, http, symbol: str) -> Optional[float]:
        """Fetch latest funding rate from Bybit."""
        try:
            resp = http.get(
                "https://api.bybit.com/v5/market/funding/history",
                params={"category": "linear", "symbol": symbol, "limit": 1}
            )
            data = resp.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                return float(data["result"]["list"][0].get("fundingRate", 0))
        except Exception as e:
            logger.debug(f"[MarketDataFetcher] Bybit funding failed for {symbol}: {e}")
        return None

    def _fetch_bybit_ls_ratio(self, http, symbol: str) -> Optional[float]:
        """Fetch long/short account ratio from Bybit."""
        try:
            resp = http.get(
                "https://api.bybit.com/v5/market/account-ratio",
                params={"category": "linear", "symbol": symbol, "period": "1h", "limit": 1}
            )
            data = resp.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                item = data["result"]["list"][0]
                buy_ratio = float(item.get("buyRatio", 0.5))
                sell_ratio = float(item.get("sellRatio", 0.5))
                return round(buy_ratio / sell_ratio, 4) if sell_ratio > 0 else 1.0
        except Exception as e:
            logger.debug(f"[MarketDataFetcher] Bybit L/S ratio failed for {symbol}: {e}")
        return None

    # ── DeFi Llama ──────────────────────────────────────────

    def fetch_defi(self):
        """
        Fetch TVL and stablecoin data from DeFi Llama.
        Completely free, no API key.
        """
        http = _get_http()
        fetched = 0

        # Total DeFi TVL
        try:
            resp = http.get("https://api.llama.fi/v2/historicalChainTvl")
            data = resp.json()
            if isinstance(data, list) and len(data) >= 2:
                latest = data[-1]
                prev = data[-2]
                tvl = float(latest.get("tvl", 0))
                prev_tvl = float(prev.get("tvl", 0))
                change_pct = ((tvl - prev_tvl) / prev_tvl * 100) if prev_tvl > 0 else 0

                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO defi_data (metric_name, value, change_pct) VALUES (?, ?, ?)",
                        ("total_tvl", tvl, round(change_pct, 3))
                    )
                    conn.commit()
                fetched += 1
                logger.info(f"[MarketDataFetcher:DeFi] Total TVL: ${tvl/1e9:.2f}B ({change_pct:+.2f}%)")
        except Exception as e:
            logger.warning(f"[MarketDataFetcher:DeFi] TVL fetch failed: {e}")

        # Stablecoin total market cap
        try:
            resp = http.get("https://stablecoins.llama.fi/stablecoins?includePrices=false")
            data = resp.json()
            if "peggedAssets" in data:
                total_mcap = sum(
                    float(s.get("circulating", {}).get("peggedUSD", 0) or 0)
                    for s in data["peggedAssets"]
                )
                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO defi_data (metric_name, value) VALUES (?, ?)",
                        ("stablecoin_mcap", total_mcap)
                    )
                    conn.commit()
                fetched += 1
                logger.info(f"[MarketDataFetcher:DeFi] Stablecoin market cap: ${total_mcap/1e9:.2f}B")
        except Exception as e:
            logger.warning(f"[MarketDataFetcher:DeFi] Stablecoin fetch failed: {e}")

        # Ethereum TVL (proxy for DeFi health)
        try:
            resp = http.get("https://api.llama.fi/v2/historicalChainTvl/Ethereum")
            data = resp.json()
            if isinstance(data, list) and len(data) >= 2:
                eth_tvl = float(data[-1].get("tvl", 0))
                prev_eth = float(data[-2].get("tvl", 0))
                change = ((eth_tvl - prev_eth) / prev_eth * 100) if prev_eth > 0 else 0

                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO defi_data (metric_name, value, change_pct) VALUES (?, ?, ?)",
                        ("ethereum_tvl", eth_tvl, round(change, 3))
                    )
                    conn.commit()
                fetched += 1
                logger.info(f"[MarketDataFetcher:DeFi] Ethereum TVL: ${eth_tvl/1e9:.2f}B ({change:+.2f}%)")
        except Exception as e:
            logger.warning(f"[MarketDataFetcher:DeFi] Ethereum TVL fetch failed: {e}")

        return fetched

    # ── FRED Macro Data ──────────────────────────────────────

    def fetch_macro(self):
        """
        Fetch macro indicators from FRED (Federal Reserve Economic Data).
        Requires FRED_API_KEY env var (free to register at fred.stlouisfed.org).
        Falls back gracefully if key unavailable.
        """
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            logger.info("[MarketDataFetcher:Macro] FRED_API_KEY not set, skipping macro data. Register free at fred.stlouisfed.org")
            return 0

        http = _get_http()
        fetched = 0

        # Series to fetch: (series_id, display_name)
        series = [
            ("DTWEXBGS", "dxy_broad"),       # Trade Weighted USD Index (DXY proxy)
            ("DGS10", "us_10y_yield"),        # 10-Year Treasury Yield
            ("DGS2", "us_2y_yield"),          # 2-Year Treasury Yield
            ("VIXCLS", "vix"),                # VIX (CBOE Volatility Index)
        ]

        for series_id, metric_name in series:
            try:
                resp = http.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 2,
                    }
                )
                data = resp.json()
                obs = data.get("observations", [])
                if not obs:
                    continue

                # Latest value
                latest = obs[0]
                val_str = latest.get("value", ".")
                if val_str == ".":
                    continue  # Missing data point
                value = float(val_str)

                # Previous value for change calculation
                prev_value = None
                change_pct = None
                if len(obs) > 1:
                    prev_str = obs[1].get("value", ".")
                    if prev_str != ".":
                        prev_value = float(prev_str)
                        change_pct = round((value - prev_value) / prev_value * 100, 4) if prev_value else None

                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO macro_data (metric_name, value, prev_value, change_pct) VALUES (?, ?, ?, ?)",
                        (metric_name, value, prev_value, change_pct)
                    )
                    conn.commit()
                fetched += 1
                change_str = f" ({change_pct:+.3f}%)" if change_pct is not None else ""
                logger.info(f"[MarketDataFetcher:Macro] {metric_name}: {value:.4f}{change_str}")

            except Exception as e:
                logger.warning(f"[MarketDataFetcher:Macro] {series_id}/{metric_name} failed: {e}")

        return fetched

    # ── yfinance Cross-Asset Data ──────────────────────────────

    def fetch_cross_asset(self):
        """
        Fetch cross-asset indicators via yfinance: DXY, VIX, S&P500, Gold.
        No API key required. Updates daily (these are traditional market hours).
        Falls back gracefully if yfinance not installed.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.info("[MarketDataFetcher:CrossAsset] yfinance not installed. Run: pip install yfinance")
            return 0

        fetched = 0
        tickers = {
            "DX-Y.NYB": "dxy",            # ICE Dollar Index (DX=F delisted on yfinance)
            "^VIX": "vix",                # CBOE Volatility Index
            "^GSPC": "sp500",             # S&P 500
            "GC=F": "gold",               # Gold Futures
            "^IXIC": "nasdaq",            # NASDAQ Composite
        }

        for symbol, metric_name in tickers.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")

                if hist.empty or len(hist) < 1:
                    logger.debug(f"[MarketDataFetcher:CrossAsset] No data for {symbol}")
                    continue

                value = float(hist['Close'].iloc[-1])
                prev_value = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else None
                change_pct = round((value - prev_value) / prev_value * 100, 4) if prev_value else None

                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO macro_data (metric_name, value, prev_value, change_pct) VALUES (?, ?, ?, ?)",
                        (metric_name, value, prev_value, change_pct)
                    )
                    conn.commit()

                fetched += 1
                change_str = f" ({change_pct:+.3f}%)" if change_pct is not None else ""
                logger.info(f"[MarketDataFetcher:CrossAsset] {metric_name}: {value:.2f}{change_str}")

            except Exception as e:
                logger.warning(f"[MarketDataFetcher:CrossAsset] {symbol}/{metric_name} failed: {e}")

        # 3-layer fallback chain: yfinance → Yahoo HTTP → FRED
        # If yfinance got SOME but not ALL, still try HTTP for the missing ones
        expected = 5  # dxy, vix, sp500, gold, nasdaq
        if fetched < expected:
            logger.info(f"[MarketDataFetcher:CrossAsset] yfinance got {fetched}/{expected}, "
                        f"trying Yahoo HTTP for missing...")
            fetched += self.fetch_cross_asset_http()

        if fetched < expected:
            logger.info(f"[MarketDataFetcher:CrossAsset] Still {fetched}/{expected}, trying FRED API...")
            fetched += self.fetch_cross_asset_fred()

        # Always try CoinGecko for BTC dominance (not available in yfinance/FRED)
        try:
            self._fetch_btc_dominance_coingecko()
        except Exception as e:
            logger.debug(f"[MarketDataFetcher:CrossAsset] CoinGecko BTC dom failed: {e}")

        return fetched

    # ── Yahoo Finance Direct HTTP (no yfinance library) ──────

    def fetch_cross_asset_http(self):
        """
        Fetch DXY, VIX, S&P500, Gold, NASDAQ via Yahoo Finance direct HTTP.
        No yfinance library needed — avoids numpy.matrix crash on numpy 2.x.
        Falls back to this when fetch_cross_asset() fails.
        """
        import requests

        fetched = 0
        tickers = {
            "DX-Y.NYB": "dxy",
            "%5EVIX": "vix",          # ^VIX URL-encoded
            "%5EGSPC": "sp500",       # ^GSPC
            "GC%3DF": "gold",         # GC=F
            "%5EIXIC": "nasdaq",      # ^IXIC
        }

        headers = {"User-Agent": "Mozilla/5.0 (compatible; FreqtradeBot/1.0)"}

        for symbol, metric_name in tickers.items():
            try:
                # Yahoo Finance v8 chart API — no auth needed
                url = (
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    f"?range=5d&interval=1d&includePrePost=false"
                )
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    logger.debug(f"[MarketDataFetcher:CrossAssetHTTP] {metric_name}: HTTP {resp.status_code}")
                    continue

                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if not result:
                    continue

                closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                # Filter None values
                closes = [c for c in closes if c is not None]
                if len(closes) < 1:
                    continue

                value = float(closes[-1])
                prev_value = float(closes[-2]) if len(closes) >= 2 else None
                change_pct = round((value - prev_value) / prev_value * 100, 4) if prev_value else None

                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO macro_data (metric_name, value, prev_value, change_pct) VALUES (?, ?, ?, ?)",
                        (metric_name, value, prev_value, change_pct)
                    )
                    conn.commit()

                fetched += 1
                change_str = f" ({change_pct:+.3f}%)" if change_pct is not None else ""
                logger.info(f"[MarketDataFetcher:CrossAssetHTTP] {metric_name}: {value:.2f}{change_str}")

            except Exception as e:
                logger.warning(f"[MarketDataFetcher:CrossAssetHTTP] {metric_name} failed: {e}")

        return fetched

    # ── FRED API (Federal Reserve — ultra reliable) ──────────

    def fetch_cross_asset_fred(self):
        """
        Fetch VIX, S&P500, Gold, NASDAQ from FRED (Federal Reserve).
        DXY only available weekly (DTWEXBGS), so less useful for daily.
        Requires FRED_API_KEY env var — free at https://fred.stlouisfed.org/docs/api/api_key.html
        """
        import requests

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            logger.info("[MarketDataFetcher:FRED] No FRED_API_KEY env var. "
                        "Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
            return 0

        fetched = 0
        # series_id → our metric_name
        series = {
            "VIXCLS": "vix",
            "SP500": "sp500",
            "GOLDAMGBD228NLBM": "gold",
            "NASDAQCOM": "nasdaq",
            "DTWEXBGS": "dxy",  # Weekly broad dollar index (best FRED has)
        }

        for series_id, metric_name in series.items():
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={api_key}&file_type=json"
                    f"&sort_order=desc&limit=5"
                )
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                observations = resp.json().get("observations", [])
                # Filter out missing values (FRED uses "." for missing)
                valid = [o for o in observations if o.get("value") not in (".", None, "")]
                if not valid:
                    continue

                value = float(valid[0]["value"])
                prev_value = float(valid[1]["value"]) if len(valid) >= 2 else None
                change_pct = round((value - prev_value) / prev_value * 100, 4) if prev_value else None

                with self._get_conn() as conn:
                    conn.execute(
                        "INSERT INTO macro_data (metric_name, value, prev_value, change_pct) VALUES (?, ?, ?, ?)",
                        (metric_name, value, prev_value, change_pct)
                    )
                    conn.commit()

                fetched += 1
                change_str = f" ({change_pct:+.3f}%)" if change_pct is not None else ""
                logger.info(f"[MarketDataFetcher:FRED] {metric_name}: {value:.2f}{change_str}")

            except Exception as e:
                logger.warning(f"[MarketDataFetcher:FRED] {series_id}/{metric_name} failed: {e}")

        return fetched

    # ── CoinGecko: BTC dominance + total crypto market cap ────

    def _fetch_btc_dominance_coingecko(self):
        """
        Fetch BTC dominance and total crypto market cap from CoinGecko.
        Free, no API key needed, 10-30 req/min rate limit.
        Evidence Engine uses btc_dom for alt vs BTC rotation signals.
        """
        import requests

        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            headers={"Accept": "application/json"},
            timeout=15
        )
        if resp.status_code != 200:
            return

        data = resp.json().get("data", {})
        btc_dom = data.get("market_cap_percentage", {}).get("btc")
        total_mcap = data.get("total_market_cap", {}).get("usd")
        mcap_change = data.get("market_cap_change_percentage_24h_usd")

        with self._get_conn() as conn:
            if btc_dom is not None:
                conn.execute(
                    "INSERT INTO macro_data (metric_name, value, change_pct) VALUES (?, ?, ?)",
                    ("btc_dominance", round(btc_dom, 2), None)
                )
                logger.info(f"[MarketDataFetcher:CoinGecko] BTC dominance: {btc_dom:.1f}%")

            if total_mcap is not None:
                conn.execute(
                    "INSERT INTO macro_data (metric_name, value, change_pct) VALUES (?, ?, ?)",
                    ("crypto_total_mcap", round(total_mcap / 1e9, 2), round(mcap_change, 2) if mcap_change else None)
                )
                logger.info(f"[MarketDataFetcher:CoinGecko] Total crypto mcap: ${total_mcap/1e9:.0f}B "
                           f"({mcap_change:+.1f}%)" if mcap_change else "")

            conn.commit()

    # ── Google Trends ──────────────────────────────────────

    def fetch_google_trends(self):
        """
        Fetch Google Trends search interest for crypto keywords.
        Detects search spikes that often precede major price moves.
        No API key required. Rate-limited by Google — use sparingly (hourly max).
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.info("[MarketDataFetcher:Trends] pytrends not installed. Run: pip install pytrends")
            return 0

        fetched = 0
        keywords = ["bitcoin", "buy bitcoin", "crypto crash", "ethereum", "crypto"]

        try:
            pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 25))
            pytrends.build_payload(keywords, cat=0, timeframe='now 7-d')
            data = pytrends.interest_over_time()

            if data.empty:
                logger.info("[MarketDataFetcher:Trends] No trend data returned")
                return 0

            # Get the latest row (most recent data point)
            latest = data.iloc[-1]

            with self._get_conn() as conn:
                for kw in keywords:
                    if kw in latest:
                        score = int(latest[kw])
                        conn.execute(
                            "INSERT INTO search_trends (keyword, interest_score) VALUES (?, ?)",
                            (kw, score)
                        )
                        fetched += 1
                conn.commit()

            # Log interesting spikes
            for kw in keywords:
                if kw in latest:
                    score = int(latest[kw])
                    avg_score = int(data[kw].mean()) if kw in data.columns else 50
                    spike = score > avg_score * 1.5
                    spike_str = " 🔥 SPIKE" if spike else ""
                    logger.info(f"[MarketDataFetcher:Trends] '{kw}': {score}/100 (avg={avg_score}){spike_str}")

        except Exception as e:
            logger.warning(f"[MarketDataFetcher:Trends] Google Trends fetch failed: {e}")

        return fetched

    # ── Aggregation / Query API ──────────────────────────────

    def get_latest_trends(self) -> Dict[str, int]:
        """Get latest Google Trends scores per keyword."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT keyword, interest_score
                    FROM search_trends
                    WHERE id IN (
                        SELECT MAX(id) FROM search_trends GROUP BY keyword
                    )
                """).fetchall()
                return {r["keyword"]: r["interest_score"] for r in rows}
        except Exception:
            return {}

    def fetch_all(self):
        """Fetch all data sources. Called by scheduler."""
        d = self.fetch_derivatives()
        f = self.fetch_defi()
        m = self.fetch_macro()
        c = self.fetch_cross_asset()
        t = self.fetch_google_trends()
        total = d + f + m + c + t
        logger.info(f"[MarketDataFetcher] fetch_all complete: {d} derivatives + {f} defi + {m} macro + {c} cross-asset + {t} trends = {total} data points")
        return total

    def get_latest_derivatives(self, pair: str = "BTC/USDT") -> Dict[str, Any]:
        """Get latest derivatives data for a pair."""
        try:
            with self._get_conn() as conn:
                row = conn.execute("""
                    SELECT * FROM derivatives_data
                    WHERE pair = ? ORDER BY timestamp DESC LIMIT 1
                """, (pair,)).fetchone()
                return dict(row) if row else {}
        except Exception:
            return {}

    def get_latest_defi(self) -> Dict[str, Any]:
        """Get latest DeFi metrics."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT metric_name, value, change_pct, timestamp
                    FROM defi_data
                    WHERE id IN (
                        SELECT MAX(id) FROM defi_data GROUP BY metric_name
                    )
                """).fetchall()
                return {r["metric_name"]: {"value": r["value"], "change_pct": r["change_pct"]} for r in rows}
        except Exception:
            return {}

    def get_latest_macro(self) -> Dict[str, Any]:
        """Get latest macro indicators."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT metric_name, value, prev_value, change_pct, timestamp
                    FROM macro_data
                    WHERE id IN (
                        SELECT MAX(id) FROM macro_data GROUP BY metric_name
                    )
                """).fetchall()
                return {r["metric_name"]: {"value": r["value"], "change_pct": r["change_pct"]} for r in rows}
        except Exception:
            return {}

    def format_for_prompt(self, pair: str = "BTC/USDT") -> str:
        """
        Format all market data as a text block for LLM prompt injection.
        Used by rag_graph.py coordinator and sentiment analyst.
        """
        parts = []

        # Derivatives
        deriv = self.get_latest_derivatives(pair)
        if deriv:
            oi = deriv.get("open_interest_usd")
            fr = deriv.get("funding_rate")
            ls = deriv.get("long_short_ratio")
            deriv_parts = []
            if oi:
                deriv_parts.append(f"OI=${oi:,.0f}")
            if fr is not None:
                fr_pct = fr * 100
                extreme = " (EXTREME)" if abs(fr) > 0.0005 else ""
                deriv_parts.append(f"Funding={fr_pct:+.4f}%{extreme}")
            if ls:
                bias = "longs dominate" if ls > 1.2 else "shorts dominate" if ls < 0.8 else "balanced"
                deriv_parts.append(f"L/S={ls:.2f} ({bias})")
            if deriv_parts:
                parts.append(f"DERIVATIVES ({pair}): {' | '.join(deriv_parts)}")

        # DeFi
        defi = self.get_latest_defi()
        if defi:
            defi_parts = []
            tvl = defi.get("total_tvl", {})
            if tvl:
                tvl_val = tvl.get("value", 0)
                tvl_chg = tvl.get("change_pct")
                chg_str = f" ({tvl_chg:+.2f}%)" if tvl_chg else ""
                defi_parts.append(f"TVL=${tvl_val/1e9:.1f}B{chg_str}")
            stable = defi.get("stablecoin_mcap", {})
            if stable:
                defi_parts.append(f"Stables=${stable.get('value', 0)/1e9:.1f}B")
            if defi_parts:
                parts.append(f"DEFI: {' | '.join(defi_parts)}")

        # Macro
        macro = self.get_latest_macro()
        if macro:
            macro_parts = []
            for name, label in [("dxy_broad", "DXY"), ("vix", "VIX"), ("us_10y_yield", "10Y"), ("us_2y_yield", "2Y")]:
                m = macro.get(name, {})
                if m:
                    val = m.get("value", 0)
                    chg = m.get("change_pct")
                    chg_str = f" ({chg:+.3f}%)" if chg else ""
                    macro_parts.append(f"{label}={val:.2f}{chg_str}")
            if macro_parts:
                parts.append(f"MACRO: {' | '.join(macro_parts)}")

        # Google Trends
        trends = self.get_latest_trends()
        if trends:
            trend_parts = []
            for kw, score in sorted(trends.items(), key=lambda x: x[1], reverse=True):
                label = "🔥" if score > 75 else "📈" if score > 50 else ""
                trend_parts.append(f'"{kw}"={score}{label}')
            if trend_parts:
                parts.append(f"SEARCH TRENDS: {' | '.join(trend_parts[:5])}")

        if not parts:
            return ""

        return "=== MARKET DATA (Phase 19 Level 3) ===\n" + "\n".join(parts)


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    fetcher = MarketDataFetcher()

    import argparse
    parser = argparse.ArgumentParser(description="Market Data Fetcher")
    parser.add_argument("--derivatives", action="store_true", help="Fetch Bybit derivatives only")
    parser.add_argument("--defi", action="store_true", help="Fetch DeFi Llama only")
    parser.add_argument("--macro", action="store_true", help="Fetch FRED macro only")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    parser.add_argument("--show", action="store_true", help="Show latest data")

    args = parser.parse_args()

    if args.show:
        print(fetcher.format_for_prompt("BTC/USDT"))
    elif args.derivatives:
        fetcher.fetch_derivatives()
    elif args.defi:
        fetcher.fetch_defi()
    elif args.macro:
        fetcher.fetch_macro()
    elif args.all:
        fetcher.fetch_all()
    else:
        parser.print_help()
