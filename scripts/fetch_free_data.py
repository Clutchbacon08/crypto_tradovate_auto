from __future__ import annotations

import os
import time
import math
import requests
import pandas as pd

OUT_CSV = os.getenv("OUT_CSV", "data/mbt_15m.csv")
SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BINANCE_INTERVAL", "1m")
LIMIT = int(os.getenv("BINANCE_LIMIT", "1000"))
BASE = os.getenv("BINANCE_BASE", "https://data-api.binance.vision")  # market-data-only endpoint
# Binance docs list /api/v3/klines; max limit 1000. :contentReference[oaicite:2]{index=2}

# How far back to fetch (minutes). Example: 180 days of 1m = 259200 minutes (too much for quick runs)
DAYS_BACK = int(os.getenv("DAYS_BACK", "60"))


def fetch_klines(start_ms: int | None, end_ms: int | None) -> list:
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    url = f"{BASE}/api/v3/klines"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    os.makedirs("data", exist_ok=True)

    # Compute start/end in ms
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - DAYS_BACK * 24 * 60 * 60 * 1000

    all_rows = []
    cur = start_ms

    # Binance returns up to LIMIT bars; step forward by last open time
    while True:
        chunk = fetch_klines(cur, end_ms)
        if not chunk:
            break

        for k in chunk:
            # kline format: [ openTime, open, high, low, close, volume, closeTime, ... ]
            all_rows.append({
                "ts_utc": pd.to_datetime(k[0], unit="ms", utc=True).isoformat(),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        last_open = chunk[-1][0]
        next_cur = last_open + 1
        if next_cur >= end_ms or next_cur <= cur:
            break
        cur = next_cur

        # gentle rate limit
        time.sleep(0.2)

    df1m = pd.DataFrame(all_rows).drop_duplicates(subset=["ts_utc"]).sort_values("ts_utc")
    if df1m.empty:
        raise SystemExit("No data fetched. Try different BASE or smaller DAYS_BACK.")

    # Resample 1m -> 15m OHLCV
    df1m["ts_utc"] = pd.to_datetime(df1m["ts_utc"], utc=True)
    df1m = df1m.set_index("ts_utc")

    df15 = pd.DataFrame({
        "open": df1m["open"].resample("15min").first(),
        "high": df1m["high"].resample("15min").max(),
        "low": df1m["low"].resample("15min").min(),
        "close": df1m["close"].resample("15min").last(),
        "volume": df1m["volume"].resample("15min").sum(),
    }).dropna()

    df15 = df15.reset_index()
    df15["ts_utc"] = df15["ts_utc"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df15.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df15)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
