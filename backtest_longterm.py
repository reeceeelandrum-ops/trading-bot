"""Long-term versions vs the live bot: rebalance less often, or just hold the whole Dow.

Momentum top-N rebalanced monthly / quarterly / yearly, and equal-weight all 30 rebalanced
yearly, vs buy-and-hold SPY, over 2016+ and 2019+. Next-open fills, 0.05% slippage.
Also reports average holding period, since gains held over a year are usually taxed less.

    .venv/bin/python backtest_longterm.py
"""
import numpy as np
import pandas as pd

import backtest as bt
from bot import load_config


def run(opens, closes, slots, every_months, lookback=126, all_stocks=False):
    mom = (closes / closes.shift(lookback) - 1).shift(1)
    cash, held, trades, buys = bt.CASH, {}, 0, {}
    holds, curve, months_seen, last_m = [], [], 0, None
    days = closes.index[lookback + 1:]
    for day in days:
        if day.month != last_m:
            last_m = day.month
            if months_seen % every_months == 0:
                o = opens.loc[day]
                pick = list(closes.columns) if all_stocks else list(mom.loc[day].dropna().sort_values(ascending=False).index[:slots])
                pick = [s for s in pick if not np.isnan(o[s])]
                for s in list(held):
                    if s not in pick:
                        cash += held.pop(s) * o[s] * (1 - bt.SLIP); trades += 1
                        holds.append((day - buys.pop(s)).days)
                value = cash + sum(q * o[k] for k, q in held.items())
                target = value / len(pick)
                for s in pick:
                    diff = target - held.get(s, 0) * o[s]
                    if abs(diff) > target * 0.25:
                        held[s] = held.get(s, 0) + diff / o[s]
                        cash -= diff + abs(diff) * bt.SLIP; trades += 1
                        buys.setdefault(s, day)
            months_seen += 1
        curve.append(cash + sum(q * closes.loc[day, k] for k, q in held.items()))
    holds += [(days[-1] - d).days for d in buys.values()]
    return pd.Series(curve, index=days), trades, np.median(holds) if holds else np.nan


def main():
    uni = load_config()["universe"]
    rows = []
    for start in ["2016-01-01", "2019-01-01"]:
        bt.START = start
        o, c = bt.download(uni + ["SPY"])
        for name, kw in [("Momentum top 10, monthly (live)", dict(slots=10, every_months=1)),
                         ("Momentum top 10, every 3 months", dict(slots=10, every_months=3)),
                         ("Momentum top 10, once a year", dict(slots=10, every_months=12)),
                         ("Momentum top 15, once a year", dict(slots=15, every_months=12)),
                         ("All 30 Dow stocks, equal, rebalanced yearly", dict(slots=30, every_months=12, all_stocks=True))]:
            eq, tr, hold = run(o[uni], c[uni], **kw)
            eq = eq[eq.index >= c.index[260]]
            s = bt.stats(name, eq / eq.iloc[0] * bt.CASH, tr, None)
            s["From"], s["Median hold"] = start[:4], f"{hold/30.4:.0f} mo"
            rows.append(s)
        spy = c["SPY"].reindex(eq.index)
        s = bt.stats("Buy and hold SPY", spy / spy.iloc[0] * bt.CASH, 1, None); s["From"], s["Median hold"] = start[:4], "forever"
        rows.append(s)
    print(pd.DataFrame(rows)[["From", "Strategy", "Per year", "Worst drop", "Return/risk", "Trades", "Median hold"]].to_string(index=False))


if __name__ == "__main__":
    main()
