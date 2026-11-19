"""Blended momentum: rank stocks by the average of their 3-, 6- and 12-month return ranks,
instead of the 6-month return alone. Compared with the live rules on both test periods.

    .venv/bin/python backtest_blend.py
"""

import numpy as np
import pandas as pd

import backtest as bt
from bot import load_config


def run_score(opens, closes, slots, score):
    """Monthly top-`slots` by `score` (known before each open), equal weight, 25% rebalance band."""
    cash, held, trades = bt.CASH, {}, 0
    curve, last_month = [], None
    days = closes.index[260:]
    for day in days:
        if day.month != last_month:
            last_month = day.month
            o = opens.loc[day]
            pick = list(score.loc[day].dropna().sort_values(ascending=False).index[:slots])
            for s in list(held):
                if s not in pick:
                    cash += held.pop(s) * o[s] * (1 - bt.SLIP); trades += 1
            value = cash + sum(q * o[k] for k, q in held.items())
            target = value / slots
            for s in pick:
                have = held.get(s, 0) * o[s]
                diff = target - have
                if abs(diff) > target * 0.25 and not np.isnan(o[s]):
                    held[s] = held.get(s, 0) + diff / o[s]
                    cash -= diff + abs(diff) * bt.SLIP; trades += 1
        curve.append(cash + sum(q * closes.loc[day, k] for k, q in held.items()))
    return pd.Series(curve, index=days), trades


def main():
    uni = load_config()["universe"]
    rows = []
    for start in ["2016-01-01", "2019-01-01"]:
        bt.START = start
        opens, closes = bt.download(uni + ["SPY"])
        c = closes[uni]
        ret = {lb: (c / c.shift(lb) - 1).shift(1) for lb in (63, 126, 252)}
        ranks = {lb: r.rank(axis=1, pct=True) for lb, r in ret.items()}
        variants = {
            "6-month (live)": ret[126],
            "Blend 3/6/12-month ranks": (ranks[63] + ranks[126] + ranks[252]) / 3,
            "Blend 6/12-month ranks": (ranks[126] + ranks[252]) / 2,
        }
        for name, score in variants.items():
            for slots in (10, 15):
                eq, tr = run_score(opens[uni], c, slots, score)
                s = bt.stats(f"{name}, top {slots}", eq / eq.iloc[0] * bt.CASH, tr, None); s["From"] = start[:4]
                rows.append(s)
        spy = closes["SPY"].reindex(eq.index)
        s = bt.stats("Buy and hold SPY", spy / spy.iloc[0] * bt.CASH, 1, None); s["From"] = start[:4]
        rows.append(s)
    print(pd.DataFrame(rows)[["From", "Strategy", "Per year", "Worst drop", "Return/risk", "Trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
