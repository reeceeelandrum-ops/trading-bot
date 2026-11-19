"""Grid test: momentum lookback x number of positions, over two start dates.

Looks for settings that work across neighbours and periods, not one lucky cell.
    .venv/bin/python backtest_grid.py
"""

import pandas as pd

import backtest as bt
from bot import load_config


def run(opens, closes, slots, lookback, skip):
    # momentum ending `skip` days ago: return from t-lookback to t-skip
    shifted = closes.shift(skip) if skip else closes
    orig = bt.run_momentum.__defaults__
    mom_closes = shifted
    # reuse run_momentum by feeding it a closes frame whose "lookback return" matches
    # the skip variant: build it directly here instead.
    import numpy as np
    mom = (mom_closes / closes.shift(lookback) - 1).shift(1)
    cash, held, trades = bt.CASH, {}, 0
    curve, last_month = [], None
    days = closes.index[lookback + 1:]
    for day in days:
        if day.month != last_month:
            last_month = day.month
            o = opens.loc[day]
            pick = list(mom.loc[day].dropna().sort_values(ascending=False).index[:slots])
            for s in list(held):
                if s not in pick:
                    cash += held.pop(s) * o[s] * (1 - bt.SLIP); trades += 1
            value = cash + sum(q * o[k] for k, q in held.items())
            target = value / slots
            for s in pick:  # rebalance every pick to equal weight
                have = held.get(s, 0) * o[s]
                diff = target - have
                if abs(diff) > target * 0.25 and not np.isnan(o[s]):
                    q = diff / o[s]
                    held[s] = held.get(s, 0) + q
                    cash -= diff + abs(diff) * bt.SLIP; trades += 1
        curve.append(cash + sum(q * closes.loc[day, k] for k, q in held.items()))
    return pd.Series(curve, index=days), trades


def main():
    cfg = load_config()
    uni = cfg["universe"]
    out = []
    for start in ["2016-01-01", "2019-01-01"]:
        bt.START = start
        opens, closes = bt.download(uni + ["SPY"])
        so, sc = opens[uni], closes[uni]
        common = None
        for name, lb, skip in [("6-month", 126, 0), ("12-month", 252, 0), ("12-1 month", 252, 21)]:
            for slots in (5, 8, 10, 15):
                eq, tr = run(so, sc, slots, lb, skip)
                eq = eq[eq.index >= sc.index[260]]  # same evaluation window for every lookback
                common = eq.index
                s = bt.stats(f"{name}, top {slots}", eq / eq.iloc[0] * bt.CASH, tr, None)
                s["From"] = start[:4]
                out.append(s)
        spy = closes["SPY"].reindex(common)
        s = bt.stats("Buy and hold SPY", spy / spy.iloc[0] * bt.CASH, 1, None); s["From"] = start[:4]
        out.append(s)
    df = pd.DataFrame(out)[["From", "Strategy", "Per year", "Worst drop", "Return/risk", "Trades"]]
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
