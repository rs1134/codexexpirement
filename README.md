# Indian Markets Dashboard (Streamlit)

A practical dashboard for:
- NSE/BSE index monitoring
- Top movers from liquid Nifty stocks
- Personal portfolio P/L tracking
- Stock-level chart + India-focused business news

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run

```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## 3) Portfolio CSV format

Upload CSV with exactly these headers:

```csv
Ticker,Quantity,Average Buy Price
RELIANCE.NS,10,2400
TCS.NS,5,3300
HDFCBANK.NS,12,1450
```

You can also enter plain symbols (like `RELIANCE`) in the stock search; the app auto-appends `.NS`.

## 4) Notes

- Data comes from Yahoo Finance via `yfinance` and may be delayed.
- If `gnewsclient` returns no ticker-specific news, the app falls back to broader Indian market business news.
- For BSE equities, use `.BO` suffix explicitly (e.g., `RELIANCE.BO`).
