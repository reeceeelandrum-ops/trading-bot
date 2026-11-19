"""Volatility-managed momentum (Barroso & Santa-Clara 2015; Daniel & Moskowitz 2016).

Each month, scale stock exposure to  min(cap, target_vol / realized_vol), where realized_vol is
the strategy's own daily-return volatility over the previous 126 trading days (annualised).
The rest sits in cash. Compared with the unscaled top-10 momentum the bot runs.

    .venv/bin/python backtest_volscale.py
"""

import numpy as np
import pandas as pd

import backtest as bt
from backtest_grid import run
from bot import load_config


def volscaled(eq: pd.Series, target: float, cap: float = 1.0) -> tuple[pd.Series, float]:
    r = eq.pct_change().fillna(0)
    vol = r.rolling(126).std() * np.sqrt(252)
    w = (target / vol).clip(upper=cap).shift(1)
    # only change exposure on the first trading day of each month (like the bot's rebalance)
    first = r.index.to_series().dt.to_period("M") != r.index.to_series().shift(1).dt.to_period("M")
    w = w.where(first).ffill().fillna(1.0)
    out = (1 + w * r).cumprod() * bt.CASH
    return out, float(w.mean())


def main():
    uni = load_config()["universe"]
    rows = []
    for start in ["2016-01-01", "2019-01-01"]:
        bt.START = start
        opens, closes = bt.download(uni + ["SPY"])
        eq, _ = run(opens[uni], closes[uni], 10, 126, 0)
        eq = eq[eq.index >= closes.index[260]]
        base = eq / eq.iloc[0] * bt.CASH
        s = bt.stats("Top-10 momentum (live bot)", base, None, 1.0); s["From"] = start[:4]; rows.append(s)
        for target in (0.12, 0.15, 0.20):
            v, avg = volscaled(base, target)
            s = bt.stats(f"Vol-managed, target {int(target*100)}%", v, None, avg); s["From"] = start[:4]; rows.append(s)
        spy = closes["SPY"].reindex(base.index)
        s = bt.stats("Buy and hold SPY", spy / spy.iloc[0] * bt.CASH, None, 1.0); s["From"] = start[:4]; rows.append(s)
    df = pd.DataFrame(rows)[["From", "Strategy", "Per year", "Worst drop", "Return/risk", "Time invested"]]
    df = df.rename(columns={"Time invested": "Avg exposure"})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
