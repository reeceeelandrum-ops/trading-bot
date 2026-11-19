"""Replay past prices to see how the bot's rules would have done.

Compares, on the same $100,000 and the same period:
  - Trend (the bot's rules): hold up to max_positions of the universe whose short
    moving average is more than buffer_pct above the long one, strongest first;
    sell when it falls more than buffer_pct below. Equal dollars per slot.
  - Momentum: each month, hold the max_positions stocks with the best 6-month return.
  - Buy and hold SPY (S&P 500 fund), and buy and hold all universe stocks equally.
  - Crypto trend on BTC and ETH vs simply holding them.

Signals use only completed days; trades happen at the next day's open, with
0.05% slippage per trade. The news, earnings and company checks can't be
replayed from price history, so they're left out.

    .venv/bin/python backtest.py            # default: since 2019
    .venv/bin/python backtest.py 2015-01-01
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

from bot import load_config

START = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
CASH = 100_000.0
SLIP = 0.0005


def download(symbols):
    d = yf.download(symbols, start=START, auto_adjust=True, progress=False)
    return d["Open"].ffill(), d["Close"].ffill()


def stats(name, equity: pd.Series, trades: int | None, invested: float | None):
    equity = equity.dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1
    dd = (equity / equity.cummax() - 1).min()
    rets = equity.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() else 0
    return {"Strategy": name, "Total return": f"{total*100:+.0f}%", "Per year": f"{cagr*100:+.1f}%",
            "Worst drop": f"{dd*100:.0f}%", "Return/risk": f"{sharpe:.2f}",
            "Trades": "—" if trades is None else trades,
            "Time invested": "—" if invested is None else f"{invested*100:.0f}%",
            "Ends at": f"${equity.iloc[-1]:,.0f}"}


def run_trend(opens, closes, short, long, buffer, slots, trading_days=252):
    s, l = closes.rolling(short).mean(), closes.rolling(long).mean()
    gap = ((s - l) / l * 100).shift(1)  # known before today's open
    cash, held, trades, inv_days = CASH, {}, 0, 0
    curve = []
    for day in closes.index[long + 1:]:
        g, o = gap.loc[day], opens.loc[day]
        # sells
        for sym in list(held):
            if g[sym] < -buffer and not np.isnan(o[sym]):
                cash += held.pop(sym) * o[sym] * (1 - SLIP); trades += 1
        # buys, strongest first
        for sym in g.dropna().sort_values(ascending=False).index:
            if len(held) >= slots or g[sym] <= buffer:
                break
            if sym in held or np.isnan(o[sym]):
                continue
            value = cash + sum(q * closes.loc[day, k] for k, q in held.items())
            spend = min(cash, value / slots)
            if spend < 1:
                break
            held[sym] = spend / (o[sym] * (1 + SLIP)); cash -= spend; trades += 1
        inv_days += bool(held)
        curve.append(cash + sum(q * closes.loc[day, k] for k, q in held.items()))
    return pd.Series(curve, index=closes.index[long + 1:]), trades, inv_days / len(curve)


def run_momentum(opens, closes, slots, lookback=126, regime=None, trail=None):
    """Monthly top-`slots` momentum. Optional:
    regime: bool Series (known before each open); when False, hold cash.
    trail:  trailing stop, e.g. 0.15 = sell if the close falls 15% below its peak since
            buying; the stock then sits out until the next monthly rebalance."""
    mom = (closes / closes.shift(lookback) - 1).shift(1)
    cash, held, peak, stopped, trades, inv = CASH, {}, {}, set(), 0, 0
    curve, last_month = [], None
    days = closes.index[lookback + 1:]
    for day in days:
        o = opens.loc[day]
        risk_on = True if regime is None else bool(regime.get(day, True))
        new_month = day.month != last_month
        if new_month:
            last_month, stopped = day.month, set()
        if not risk_on and held:
            for sym in list(held):
                cash += held.pop(sym) * o[sym] * (1 - SLIP); trades += 1
        if trail is not None:
            prev = closes.shift(1).loc[day]
            for sym in list(held):
                peak[sym] = max(peak.get(sym, 0), prev[sym])
                if prev[sym] < peak[sym] * (1 - trail):
                    cash += held.pop(sym) * o[sym] * (1 - SLIP); trades += 1; stopped.add(sym)
        if risk_on and (new_month or len(held) < slots):
            pick = [s for s in mom.loc[day].dropna().sort_values(ascending=False).index[:slots]]
            if new_month:
                for sym in list(held):
                    if sym not in pick:
                        cash += held.pop(sym) * o[sym] * (1 - SLIP); trades += 1
            value = cash + sum(q * o[k] for k, q in held.items())
            for sym in pick:
                if sym not in held and sym not in stopped and not np.isnan(o[sym]):
                    spend = min(cash, value / slots)
                    if spend < 1:
                        break
                    held[sym] = spend / (o[sym] * (1 + SLIP)); cash -= spend; trades += 1
                    peak[sym] = o[sym]
        inv += bool(held)
        curve.append(cash + sum(q * closes.loc[day, k] for k, q in held.items()))
    return pd.Series(curve, index=days), trades, inv / len(days)


def main():
    cfg = load_config()
    uni, short, long, buf, slots = cfg["universe"], cfg["short_window"], cfg["long_window"], cfg["buffer_pct"], cfg["max_positions"]
    print(f"Backtest from {START}. Universe: {len(uni)} stocks. Rules: SMA{short}/SMA{long}, "
          f"buffer ±{buf}%, up to {slots} positions, next-day open fills, {SLIP*100:.2f}% slippage.\n")

    opens, closes = download(uni + ["SPY"])
    so, sc = opens[uni], closes[uni]
    rows = []

    eq, tr, inv = run_trend(so, sc, short, long, buf, slots)
    common = eq.index
    rows.append(stats(f"Trend, current rules (±{buf}% buffer)", eq, tr, inv))
    for b in (0.0, 3.0):
        e2, t2, i2 = run_trend(so, sc, short, long, b, slots)
        rows.append(stats(f"Trend, ±{b:g}% buffer", e2.reindex(common), t2, i2))
    e3, t3, i3 = run_trend(so, sc, short, long, buf, 10)
    rows.append(stats("Trend, 10 positions", e3.reindex(common), t3, i3))
    em, tm, im = run_momentum(so, sc, slots)
    rows.append(stats(f"Momentum, top {slots} monthly (live bot)", em.reindex(common), tm, im))
    # Ideas from the second-brain wiki (Murphy, golden-cross evidence, position-sizing notes)
    spy_c = closes["SPY"]
    golden = (spy_c.rolling(50).mean() > spy_c.rolling(200).mean()).shift(1)
    above200 = (spy_c > spy_c.rolling(200).mean()).shift(1)
    for label, kw in [("+ SPY golden-cross filter", {"regime": golden}),
                      ("+ SPY above 200-day filter", {"regime": above200}),
                      ("+ 15% trailing stop", {"trail": 0.15}),
                      ("+ 25% trailing stop", {"trail": 0.25}),
                      ("+ golden-cross filter + 15% stop", {"regime": golden, "trail": 0.15})]:
        e, t, i = run_momentum(so, sc, slots, **kw)
        rows.append(stats(f"Momentum {label}", e.reindex(common), t, i))
    e10, t10, i10 = run_momentum(so, sc, 10)
    rows.append(stats("Momentum, top 10 monthly", e10.reindex(common), t10, i10))
    spy = closes["SPY"].reindex(common)
    rows.append(stats("Buy and hold SPY", spy / spy.iloc[0] * CASH, 1, 1.0))
    ew = (sc.reindex(common) / sc.reindex(common).iloc[0]).mean(axis=1)
    rows.append(stats("Buy and hold all 30 equally", ew * CASH, 30, 1.0))
    print("STOCKS")
    print(pd.DataFrame(rows).to_string(index=False))

    co, cc = download(["BTC-USD", "ETH-USD"])
    crows = []
    for sym in ["BTC-USD", "ETH-USD"]:
        e, t, i = run_trend(co[[sym]], cc[[sym]], short, long, buf, 1, 365)
        crows.append(stats(f"{sym[:3]} trend 20/50 (live bot)", e, t, i))
        e2, t2, i2 = run_trend(co[[sym]], cc[[sym]], 50, 200, buf, 1, 365)
        crows.append(stats(f"{sym[:3]} golden cross 50/200", e2.reindex(e.index), t2, i2))
        p = cc[sym].reindex(e.index)
        crows.append(stats(f"{sym[:3]} buy and hold", p / p.iloc[0] * CASH, 1, 1.0))
    print("\nCRYPTO")
    print(pd.DataFrame(crows).to_string(index=False))
    print("\nCaveats: the 30 stocks are today's Dow members, which flatters every stock strategy "
          "(companies that fell out of the index aren't included). Past results don't predict future ones.")


if __name__ == "__main__":
    main()
