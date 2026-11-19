"""Intraday momentum test (Gao, Han, Li & Zhou 2018, JFE): does SPY's first half-hour
(previous close -> 10:00 ET) predict its last half-hour (15:30 -> 16:00)?

Strategy tested: at 15:30 ET, if the first half-hour return was positive, buy SPY and sell at
the close; if negative, stay out (long-only) or short (long-short, for reference only - the bot
doesn't short). Costs: 0.02% per side. Uses Alpaca 30-minute bars (IEX feed).

    .venv/bin/python intraday_test.py
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bot import HERE, NY

COST = 0.0002


def bars(symbol, start):
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    load_dotenv(HERE / ".env")
    c = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                                           start=start, feed=DataFeed.IEX)).df.loc[symbol]
    df.index = df.index.tz_convert(NY)
    return df


def main():
    df = bars("SPY", datetime(2016, 1, 1, tzinfo=timezone.utc))
    df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time < pd.Timestamp("16:00").time())]
    rows = []
    prev_close = None
    for day, g in df.groupby(df.index.date):
        g = g.sort_index()
        t = {ts.strftime("%H:%M"): r for ts, r in g.iterrows()}
        if prev_close is not None and "09:30" in t and "15:30" in t:
            r_first = t["09:30"].close / prev_close - 1        # prev close -> 10:00
            r_last = t["15:30"].close / t["15:30"].open - 1    # 15:30 -> 16:00
            rows.append({"day": pd.Timestamp(day), "first": r_first, "last": r_last})
        if "15:30" in t:
            prev_close = t["15:30"].close
    d = pd.DataFrame(rows).set_index("day")
    print(f"SPY days tested: {len(d)}  ({d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d})")
    corr = d["first"].corr(d["last"])
    print(f"Correlation, first half-hour vs last half-hour: {corr:+.3f}")

    long_only = np.where(d["first"] > 0, d["last"] - 2 * COST, 0.0)
    long_short = np.sign(d["first"]) * d["last"] - 2 * COST
    always = d["last"] - 2 * COST
    for name, r in [("Long-only (buy at 15:30 if first half-hour up)", long_only),
                    ("Long-short (reference only)", long_short),
                    ("Always buy the last half-hour", always)]:
        r = pd.Series(r, index=d.index)
        traded = r[r != 0]
        eq = (1 + r).cumprod()
        yrs = len(d) / 252
        print(f"\n{name}")
        print(f"  total {eq.iloc[-1]-1:+.1%}, per year {eq.iloc[-1]**(1/yrs)-1:+.2%}, worst drop {(eq/eq.cummax()-1).min():.1%}")
        print(f"  days traded {len(traded)}, win rate {(traded>0).mean():.0%}, avg per trade {traded.mean()*100:+.3f}%")
        by_year = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        print("  by year: " + ", ".join(f"{y}: {v:+.1%}" for y, v in by_year.items()))


if __name__ == "__main__":
    main()
