"""Does a bigger stock pool help momentum? Dow 30 vs S&P 100 (today's members, so both are
flattered by survivorship - the S&P 100 more so, since it includes recent winners added late).

    .venv/bin/python backtest_universe.py
"""
import numpy as np
import pandas as pd

import backtest as bt
from backtest_grid import run
from bot import load_config

SP100 = ("AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK BMY BRK-B C CAT "
         "CHTR CL CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR F FDX GD GE GILD GM GOOGL GS "
         "HD HON IBM INTC INTU ISRG JNJ JPM KHC KO LIN LLY LMT LOW MA MCD MDLZ MDT MET META MMM MO MRK "
         "MS MSFT NEE NFLX NKE NOW NVDA ORCL PEP PFE PG PM PYPL QCOM RTX SBUX SCHW SO SPG T TGT TMO TMUS "
         "TSLA TXN UNH UNP UPS USB V VZ WFC WMT XOM").split()


def main():
    dow = load_config()["universe"]
    rows = []
    for start in ["2016-01-01", "2019-01-01"]:
        bt.START = start
        opens, closes = bt.download(sorted(set(SP100 + dow + ["SPY"])))
        for name, uni in [("Dow 30", dow), ("S&P 100", [s for s in SP100 if s in closes.columns])]:
            for slots in (10, 15, 20):
                eq, tr = run(opens[uni], closes[uni], slots, 126, 0)
                eq = eq[eq.index >= closes.index[260]]
                s = bt.stats(f"{name}, top {slots}", eq / eq.iloc[0] * bt.CASH, tr, None); s["From"] = start[:4]
                rows.append(s)
        spy = closes["SPY"].reindex(eq.index)
        s = bt.stats("Buy and hold SPY", spy / spy.iloc[0] * bt.CASH, 1, None); s["From"] = start[:4]
        rows.append(s)
    print(pd.DataFrame(rows)[["From", "Strategy", "Per year", "Worst drop", "Return/risk", "Trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
