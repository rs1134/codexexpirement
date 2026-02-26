"""Streamlit dashboard for Indian markets, portfolio tracking, and finance news.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from gnewsclient import gnewsclient
except Exception:  # pragma: no cover - optional dependency import guard
    gnewsclient = None

# ------------------------------- App Configuration -------------------------------
st.set_page_config(
    page_title="Indian Markets Dashboard",
    page_icon="📈",
    layout="wide",
)

# Dark-theme-friendly style adjustments for metric cards and table areas.
st.markdown(
    """
    <style>
        .stApp {background-color: #0E1117; color: #FAFAFA;}
        [data-testid="stMetricValue"] {color: #F8F8F2;}
        [data-testid="stMetricDelta"] svg {display: inline;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Constants -------------------------------
INDEX_TICKERS: Dict[str, str] = {
    "Nifty 50": "^NSEI",
    "BSE Sensex": "^BSESN",
}

# High-volume representatives from Nifty 50 universe for movers table.
NIFTY50_LIQUID_STOCKS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "ASIANPAINT.NS",
]

DEFAULT_PORTFOLIO = pd.DataFrame(
    [
        {"Ticker": "RELIANCE.NS", "Quantity": 10, "Average Buy Price": 2400},
        {"Ticker": "TCS.NS", "Quantity": 5, "Average Buy Price": 3300},
        {"Ticker": "HDFCBANK.NS", "Quantity": 12, "Average Buy Price": 1450},
        {"Ticker": "INFY.NS", "Quantity": 8, "Average Buy Price": 1500},
    ]
)


def normalize_indian_ticker(raw_ticker: str) -> str:
    """Normalize user ticker input and append `.NS` for NSE equities when missing.

    Rules:
    - Keep index symbols (like ^NSEI) untouched.
    - Keep explicitly suffixed tickers (.NS/.BO) untouched.
    - For plain equity symbols (e.g., RELIANCE), append .NS by default.
    """
    ticker = raw_ticker.strip().upper()
    if not ticker:
        return ticker
    if ticker.startswith("^"):
        return ticker
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    return f"{ticker}.NS"


# ------------------------------- Data Functions -------------------------------
@st.cache_data(ttl=120)
def fetch_history(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data for a single ticker using yfinance.

    Raises:
        ValueError: If no data is returned for the requested ticker/period.
    """
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")
    return df


@st.cache_data(ttl=120)
def fetch_latest_quote_and_change(ticker: str) -> Dict[str, float]:
    """Return latest close and day-over-day percentage change from 2-day history."""
    hist = fetch_history(ticker, period="5d", interval="1d")
    close_series = hist["Close"].dropna()
    if len(close_series) < 2:
        raise ValueError(f"Insufficient data to compute daily change for {ticker}")

    current = float(close_series.iloc[-1])
    prev = float(close_series.iloc[-2])
    pct_change = ((current - prev) / prev) * 100 if prev else 0.0
    return {"price": current, "pct_change": pct_change}


@st.cache_data(ttl=300)
def fetch_top_movers(tickers: List[str], top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate intraday-ish movers using the latest two daily closes.

    The function uses 2-day/1-day history as a practical approximation for current
    trading day movement in a free data environment.
    """
    rows = []
    for ticker in tickers:
        try:
            hist = fetch_history(ticker, period="5d", interval="1d")
            close = hist["Close"].dropna()
            if len(close) < 2:
                continue
            last_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            pct_change = ((last_price - prev_close) / prev_close) * 100 if prev_close else 0.0
            rows.append(
                {
                    "Ticker": ticker,
                    "Last Price": round(last_price, 2),
                    "Change %": round(pct_change, 2),
                }
            )
        except Exception:
            continue

    movers = pd.DataFrame(rows)
    if movers.empty:
        return movers, movers

    gainers = movers.sort_values("Change %", ascending=False).head(top_n).reset_index(drop=True)
    losers = movers.sort_values("Change %", ascending=True).head(top_n).reset_index(drop=True)
    return gainers, losers


@st.cache_data(ttl=180)
def fetch_last_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Get latest close prices for a list of tickers with graceful failure."""
    prices: Dict[str, Optional[float]] = {}
    for ticker in tickers:
        try:
            hist = fetch_history(ticker, period="5d", interval="1d")
            last_price = float(hist["Close"].dropna().iloc[-1])
            prices[ticker] = last_price
        except Exception:
            prices[ticker] = None
    return prices


@st.cache_data(ttl=900)
def fetch_news(query: str, max_items: int = 5) -> List[Dict[str, str]]:
    """Fetch business news focused on India using gnewsclient.

    Fallback behavior:
    - If query-specific search returns nothing, fetch general Indian business news.
    - If gnewsclient is unavailable/fails, return an empty list.
    """
    if gnewsclient is None:
        return []

    try:
        client = gnewsclient.NewsClient(
            language="english",
            location="india",
            topic="Business",
            max_results=max_items,
        )
        articles = client.get_news(search=query) or []
        if not articles:
            articles = client.get_news() or []

        # Normalize keys for stable rendering.
        normalized = []
        for item in articles[:max_items]:
            normalized.append(
                {
                    "title": item.get("title", "Untitled"),
                    "link": item.get("link", ""),
                    "publisher": item.get("publisher", "Unknown"),
                }
            )
        return normalized
    except Exception:
        return []


# ------------------------------- Portfolio Logic -------------------------------
def parse_portfolio_csv(uploaded_file) -> pd.DataFrame:
    """Parse uploaded CSV and validate required columns."""
    raw = uploaded_file.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(raw))

    required_cols = {"Ticker", "Quantity", "Average Buy Price"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain: Ticker, Quantity, Average Buy Price")

    df = df[["Ticker", "Quantity", "Average Buy Price"]].copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Buy Price"] = pd.to_numeric(df["Average Buy Price"], errors="coerce")

    if df[["Quantity", "Average Buy Price"]].isna().any().any():
        raise ValueError("Quantity and Average Buy Price must be numeric.")

    return df


def calculate_portfolio_metrics(holdings_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Compute per-asset and aggregate portfolio performance metrics."""
    df = holdings_df.copy()
    prices = fetch_last_prices(df["Ticker"].tolist())
    df["Current Price"] = df["Ticker"].map(prices)
    df = df.dropna(subset=["Current Price"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid tickers found in portfolio.")

    df["Investment Value"] = df["Quantity"] * df["Average Buy Price"]
    df["Current Value"] = df["Quantity"] * df["Current Price"]
    df["Unrealized P/L"] = df["Current Value"] - df["Investment Value"]
    df["Unrealized P/L %"] = (df["Unrealized P/L"] / df["Investment Value"]) * 100

    totals = {
        "investment": float(df["Investment Value"].sum()),
        "current": float(df["Current Value"].sum()),
        "pl": float(df["Unrealized P/L"].sum()),
    }
    totals["pl_pct"] = (totals["pl"] / totals["investment"] * 100) if totals["investment"] else 0.0

    return df, totals


# ------------------------------- UI Rendering -------------------------------
st.title("🇮🇳 Indian Financial Markets Dashboard")
st.caption("Track NSE/BSE indices, top movers, personal portfolio, and stock-specific news.")

with st.sidebar:
    st.header("How to Use")
    st.markdown(
        """
        1. **Market Overview** → monitor Nifty 50 and Sensex trend snapshots.
        2. **Top Movers** → quick list of gainers/losers from liquid Nifty names.
        3. **Personal Portfolio** → upload CSV (`Ticker,Quantity,Average Buy Price`) or use sample data.
        4. **Stock Detail & News** → search `RELIANCE`, `TCS.NS`, or `^NSEI`.

        Tip: This app defaults to **NSE** by appending `.NS` when you enter plain symbols.
        """
    )

market_tab, movers_tab, portfolio_tab, detail_news_tab = st.tabs(
    ["Market Overview", "Top Movers", "Personal Portfolio", "Stock Detail & News"]
)

with market_tab:
    st.subheader("Market Overview: Nifty 50 & Sensex")
    cols = st.columns(2)

    for idx, (name, ticker) in enumerate(INDEX_TICKERS.items()):
        with cols[idx]:
            try:
                quote = fetch_latest_quote_and_change(ticker)
                st.metric(
                    label=f"{name} ({ticker})",
                    value=f"{quote['price']:.2f}",
                    delta=f"{quote['pct_change']:.2f}%",
                )

                hist_1m = fetch_history(ticker, period="1mo", interval="1d")
                mini = go.Figure()
                mini.add_trace(
                    go.Scatter(
                        x=hist_1m.index,
                        y=hist_1m["Close"],
                        mode="lines",
                        name=name,
                        line=dict(width=2),
                    )
                )
                mini.update_layout(
                    height=240,
                    margin=dict(l=10, r=10, t=20, b=10),
                    template="plotly_dark",
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=False,
                )
                st.plotly_chart(mini, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not load {name} data: {exc}")

with movers_tab:
    st.subheader("Top 5 Gainers & Losers (from liquid Nifty 50 stocks)")
    try:
        gainers_df, losers_df = fetch_top_movers(NIFTY50_LIQUID_STOCKS, top_n=5)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🚀 Top Gainers")
            if gainers_df.empty:
                st.info("No movers data available right now.")
            else:
                st.dataframe(gainers_df, use_container_width=True)
        with c2:
            st.markdown("#### 📉 Top Losers")
            if losers_df.empty:
                st.info("No movers data available right now.")
            else:
                st.dataframe(losers_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to fetch movers: {exc}")

with portfolio_tab:
    st.subheader("Personal Portfolio Tracker")
    st.write("Upload a CSV or use sample holdings.")
    st.code("Ticker,Quantity,Average Buy Price", language="text")

    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    use_default = st.checkbox("Use sample portfolio", value=uploaded is None)

    holdings: Optional[pd.DataFrame] = None
    if uploaded is not None and not use_default:
        try:
            holdings = parse_portfolio_csv(uploaded)
            st.success("Portfolio CSV loaded successfully.")
        except Exception as exc:
            st.error(f"CSV parsing error: {exc}")
    elif use_default:
        holdings = DEFAULT_PORTFOLIO.copy()

    if holdings is not None:
        try:
            metrics_df, totals = calculate_portfolio_metrics(holdings)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Investment", f"₹{totals['investment']:,.2f}")
            m2.metric("Current Value", f"₹{totals['current']:,.2f}")
            m3.metric("Unrealized P/L", f"₹{totals['pl']:,.2f}", delta=f"{totals['pl_pct']:.2f}%")

            show_df = metrics_df.copy()
            show_df = show_df[
                [
                    "Ticker",
                    "Quantity",
                    "Average Buy Price",
                    "Current Price",
                    "Investment Value",
                    "Current Value",
                    "Unrealized P/L",
                    "Unrealized P/L %",
                ]
            ]
            st.dataframe(show_df.round(2), use_container_width=True)

            pie = px.pie(
                metrics_df,
                names="Ticker",
                values="Current Value",
                title="Portfolio Allocation by Current Value",
                template="plotly_dark",
            )
            st.plotly_chart(pie, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not calculate portfolio metrics: {exc}")

with detail_news_tab:
    st.subheader("Stock Detail & Related News")
    ticker_input = st.text_input("Enter Indian stock ticker", value="RELIANCE.NS")

    if ticker_input:
        ticker_input = normalize_indian_ticker(ticker_input)
        col_chart, col_news = st.columns([2, 1])

        with col_chart:
            st.markdown(f"#### 6-Month Candlestick: {ticker_input}")
            try:
                cdf = fetch_history(ticker_input, period="6mo", interval="1d")
                candle = go.Figure(
                    data=[
                        go.Candlestick(
                            x=cdf.index,
                            open=cdf["Open"],
                            high=cdf["High"],
                            low=cdf["Low"],
                            close=cdf["Close"],
                            name=ticker_input,
                        )
                    ]
                )
                candle.update_layout(
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(candle, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not load stock chart for {ticker_input}: {exc}")

        with col_news:
            st.markdown("#### Latest News")
            # Try ticker symbol then fallback to broader Indian market phrase.
            symbol_query = ticker_input.split(".")[0]
            news_items = fetch_news(symbol_query, max_items=5)
            if not news_items:
                news_items = fetch_news("Indian stock market", max_items=5)

            if not news_items:
                st.warning("No news available. Install/configure gnewsclient or retry later.")
            else:
                for idx, item in enumerate(news_items, start=1):
                    title = item.get("title", "Untitled")
                    link = item.get("link", "")
                    publisher = item.get("publisher", "Unknown")
                    if link:
                        st.markdown(f"{idx}. [{title}]({link})  ")
                    else:
                        st.markdown(f"{idx}. {title}  ")
                    st.caption(f"Source: {publisher}")
