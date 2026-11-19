"""Grade the early-event scanner's logged predictions against SPY.

Each prediction is "entered" at the open of the first session after it was discovered and
measured at the close of its horizon-th session, minus SPY over exactly the same window, signed
by the predicted direction (up = +1, down = -1). Nothing is traded.

    .venv/bin/python scanner_eval.py "<path to predictions.csv>"          # grade due rows, print record
    .venv/bin/python scanner_eval.py "<path>" --report-only               # just print the record
"""

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

COLUMNS = ["id", "discovered_at_utc", "asset", "direction", "stage", "score", "horizon_days",
           "price_at_discovery", "source_url", "event", "invalidation", "status",
           "asset_return", "spy_return", "excess_in_direction", "graded_at_utc"]


def grade(discovered: pd.Timestamp, horizon: int, direction: str,
          asset_open: pd.Series, asset_close: pd.Series,
          spy_open: pd.Series, spy_close: pd.Series):
    """Return (asset_return, spy_return, excess_in_direction) or None if the horizon hasn't ended.
    Series are daily, indexed by session date (tz-naive). A session counts only if its date is
    after the discovery date (discoveries during or after a session enter at the next open)."""
    sessions = [d for d in asset_close.dropna().index if d.date() > discovered.date()]
    sessions = [d for d in sessions if d in spy_close.index]
    if len(sessions) < horizon:
        return None
    start, end = sessions[0], sessions[horizon - 1]
    a = asset_close[end] / asset_open[start] - 1
    s = spy_close[end] / spy_open[start] - 1
    sign = 1 if str(direction).strip().lower() == "up" else -1
    return a, s, sign * (a - s)


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).reindex(columns=COLUMNS)
    return df


def record(df: pd.DataFrame) -> str:
    g = df[df.status == "graded"].copy()
    lines = [f"Predictions logged: {len(df)}  |  graded: {len(g)}  |  open: {(df.status == 'open').sum()}"]
    if len(g):
        g["x"] = g.excess_in_direction.astype(float)
        lines.append(f"Average excess vs SPY (in predicted direction): {g.x.mean()*100:+.2f}%  |  hit rate: {(g.x > 0).mean():.0%}")
        for stage, s in g.groupby("stage"):
            lines.append(f"  {stage}: n={len(s)}, avg {s.x.mean()*100:+.2f}%, hit {(s.x > 0).mean():.0%}")
        if len(g) >= 30:
            avg, hit = g.x.mean(), (g.x > 0).mean()
            verdict = ("PROMISING (see playbook: also check each stage)" if avg > 0.01 and hit > 0.55
                       else "RETIRE" if avg <= 0 or hit <= 0.50 else "KEEP LOGGING to 60")
            lines.append(f"Verdict vs pre-set criteria: {verdict}")
        else:
            lines.append(f"Verdict: too early ({len(g)}/30 graded predictions needed)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    df = load(args.ledger)
    due = df[(df.status == "open")]
    if not args.report_only and len(due):
        import yfinance as yf
        assets = sorted(set(due.asset.str.strip()))
        start = (pd.to_datetime(due.discovered_at_utc).min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        px = yf.download(assets + ["SPY"], start=start, auto_adjust=True, progress=False, threads=False)
        o, c = px["Open"], px["Close"]
        for i, r in due.iterrows():
            sym = r.asset.strip()
            if sym not in c.columns:
                print(f"  {r.id}: no price data for {sym}", file=sys.stderr)
                continue
            res = grade(pd.Timestamp(r.discovered_at_utc).tz_localize(None) if pd.Timestamp(r.discovered_at_utc).tzinfo is None
                        else pd.Timestamp(r.discovered_at_utc).tz_convert("America/New_York").tz_localize(None),
                        int(float(r.horizon_days)), r.direction, o[sym], c[sym], o["SPY"], c["SPY"])
            if res:
                a, s, x = res
                df.loc[i, ["status", "asset_return", "spy_return", "excess_in_direction", "graded_at_utc"]] = [
                    "graded", f"{a:.5f}", f"{s:.5f}", f"{x:.5f}", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")]
                print(f"  graded {r.id} {sym} {r.direction}: asset {a*100:+.2f}%, SPY {s*100:+.2f}%, excess {x*100:+.2f}%")
        df.to_csv(args.ledger, index=False)
    print(record(df))


if __name__ == "__main__":
    main()
