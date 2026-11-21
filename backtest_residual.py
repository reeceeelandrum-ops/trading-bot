"""Residual (beta-adjusted) momentum: rank stocks on the part of their return the market doesn't
explain, instead of on raw 6-month return. Compared with the live rules and SPY on both test periods.

Source: Blitz, Huij & Martens (2011), "Residual momentum", Journal of Empirical Finance 18(3):
ranking on residuals from a factor regression roughly doubled risk-adjusted momentum profits and
held in large caps. Their version uses Fama-French 3 factors on monthly data; here the only factor
is SPY (market model) on daily data, the data the bot already has.

At each month's first trading day (signals from the previous close, fills at that day's open):
  1. For each stock, fit r_stock = a + b * r_SPY on the last `est` trading days.
  2. Residual = r_stock - a - b * r_SPY, summed over the formation window
     (last `form` days, optionally skipping the most recent `skip` days).
  3. Score = sum of residuals / their standard deviation (BHM's scaling), or the raw sum.
Top 10 by score, equal weight, 25% band — the same engine as backtest_blend.py.

    .venv/bin/python backtest_residual.py
"""

import numpy as np
import pandas as pd

import backtest as bt
import backtest_blend as bb
from bot import load_config

WARMUP = 260  # run_score starts trading after this many rows


def residual_scores(closes: pd.DataFrame, spy: pd.Series, est: int, form: int, skip: int, scaled: bool):
    """Score for each month's first trading day, using closes up to the previous day only."""
    rets = closes.pct_change()
    m = spy.pct_change()
    idx = closes.index
    out = pd.DataFrame(np.nan, index=idx, columns=closes.columns)
    month_starts = [i for i in range(1, len(idx)) if idx[i].month != idx[i - 1].month]
    for i in month_starts:
        end = i  # rows < i are known before day i's open
        if end - est < 1:
            continue
        R = rets.iloc[end - est:end]
        M = m.iloc[end - est:end].values
        X = np.column_stack([np.ones_like(M), M])
        for s in closes.columns:
            y = R[s].values
            ok = ~np.isnan(y) & ~np.isnan(M)
            if ok.sum() < est * 0.8:
                continue
            coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            resid = y - X @ coef
            window = resid[-form:len(resid) - skip if skip else None]
            window = window[~np.isnan(window)]
            if len(window) < 20:
                continue
            out.iloc[i, out.columns.get_loc(s)] = window.sum() / window.std() if scaled else window.sum()
    # run_score only reads month-start rows; fill forward so any row lookup is defined
    return out.ffill()


def main():
    uni = load_config()["universe"]  # live investable list (hands_off removed)
    bt.START = "2012-01-01"  # warm-up for 3-year beta windows
    opens, closes = bt.download(uni + ["SPY"])
    c, spy = closes[uni], closes["SPY"]
    live = (c / c.shift(126) - 1).shift(1)

    variants = {"6-month raw return (live)": live}
    grid = [  # (estimation days, formation days, skip days, scaled)
        (756, 126, 0, True),   # 3-yr beta, 6-month residual, scaled  — main hypothesis
        (252, 126, 0, True),   # 1-yr beta
        (756, 252, 21, True),  # BHM's 12-1 month formation
        (756, 126, 0, False),  # unscaled residual sum
        (252, 252, 21, True),
    ]
    for est, form, skip, scaled in grid:
        name = f"Residual {form}d{'-' + str(skip) if skip else ''}, beta {est // 252}y{'' if scaled else ', unscaled'}"
        variants[name] = residual_scores(c, spy, est, form, skip, scaled)

    rows = []
    for start in ["2016-01-01", "2019-01-01"]:
        p0 = c.index.searchsorted(pd.Timestamp(start))
        sl = slice(max(p0 - WARMUP, 0), None)
        for name, score in variants.items():
            eq, tr = bb.run_score(opens[uni].iloc[sl], c.iloc[sl], 10, score.iloc[sl])
            eq = eq.loc[start:]
            s = bt.stats(name, eq / eq.iloc[0] * bt.CASH, tr, None); s["From"] = start[:4]
            rows.append(s)
        sp = spy.loc[eq.index]
        s = bt.stats("Buy and hold SPY", sp / sp.iloc[0] * bt.CASH, 1, None); s["From"] = start[:4]
        rows.append(s)
    print(pd.DataFrame(rows)[["From", "Strategy", "Per year", "Worst drop", "Return/risk", "Trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
