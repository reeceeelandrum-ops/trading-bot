"""Does the bot's keyword news score predict returns? Replays it on Alpaca's historical news.

Alpaca's news history on this plan only reaches back ~7 months, so treat results as a first
look, not proof. Headlines are cached in .cache/news/ (gitignored).

1. Event study: after a stock's 3-day score reaches news_sell_score (or news_block_buy_score),
   what are its next 1/5/20-day returns vs. the same stock's average?
2. Replay: top-10 monthly momentum with vs. without the news sell, rebound buy-back and
   news buy-block, over the news window.

    .venv/bin/python news_backtest.py
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from bot import HERE, NY, load_config
from signals import headline_score

CACHE = HERE / ".cache" / "news"
DAYS = 215


def fetch_news(symbols):
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    load_dotenv(HERE / ".env")
    nc = NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    CACHE.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc) - timedelta(days=DAYS)
    rows = []
    for sym in symbols:
        f = CACHE / f"{sym}.json"
        if not f.exists():
            items = nc.get_news(NewsRequest(symbols=sym, start=start, limit=None)).data.get("news", [])
            f.write_text(json.dumps([{"t": n.created_at.isoformat(), "h": n.headline, "n": len(n.symbols)} for n in items]))
        for n in json.loads(f.read_text()):
            if n["n"] <= 3:  # same filter the bot uses: stories mainly about this company
                rows.append({"symbol": sym, "time": pd.Timestamp(n["t"]), "headline": n["h"]})
    df = pd.DataFrame(rows)
    df["score"] = df.headline.map(headline_score)
    # A headline counts for the next trading session if published before that day's open.
    t = df.time.dt.tz_convert(NY)
    df["day"] = (t + pd.Timedelta(hours=14, minutes=30)).dt.normalize().dt.tz_localize(None)  # after 9:30 -> next day
    return df


def daily_scores(news, index, symbols, window=3):
    """3-calendar-day rolling sum of headline scores known before each day's open."""
    s = news.groupby(["day", "symbol"]).score.sum().unstack(fill_value=0)
    full = pd.date_range(index.min() - pd.Timedelta(days=10), index.max(), freq="D")
    s = s.reindex(full, fill_value=0).reindex(columns=symbols, fill_value=0)
    return s.rolling(window, min_periods=1).sum().reindex(index)


def event_study(score, opens, closes, level, label):
    out = []
    for h in (1, 5, 20):
        fwd = closes.shift(-h) / opens - 1          # buy/sell at today's open, hold h days
        base = fwd.stack().dropna().mean()
        hit = fwd[score <= level].stack().dropna()
        out.append({"Signal": label, "Days after": h, "Events": len(hit),
                    "Avg return after signal": f"{hit.mean()*100:+.2f}%",
                    "Avg return any day": f"{base*100:+.2f}%",
                    "Difference": f"{(hit.mean()-base)*100:+.2f} pts",
                    "Share that fell": f"{(hit < 0).mean()*100:.0f}%"})
    return out


def replay(opens, closes, score, cfg, use_news):
    """Top-10 monthly momentum, daily steps, optional news rules (as in bot.py)."""
    slots, lb = cfg["max_positions"], cfg["momentum_lookback_days"]
    mom = (closes / closes.shift(lb) - 1)
    cash, held, sold, trades = 100_000.0, {}, {}, 0
    curve, month, targets = [], None, []
    for day in closes.index:
        if day < score.index.min():
            continue
        o = opens.loc[day]
        if day.month != month:
            month = day.month
            prev = mom[mom.index < day.replace(day=1)]
            targets = list(prev.iloc[-1].dropna().sort_values(ascending=False).index[:slots])
            for s in list(held):
                if s not in targets:
                    cash += held.pop(s) * o[s]; trades += 1
        sc = score.loc[day]
        if use_news:
            for s in list(held):
                if sc[s] <= cfg["news_sell_score"]:
                    cash += held[s] * o[s]; sold[s] = (o[s], day); held.pop(s); trades += 1
        value = cash + sum(q * o[k] for k, q in held.items())
        size = value / slots
        for s in targets:
            if s in held or np.isnan(o[s]):
                continue
            if use_news and s in sold:
                px, when = sold[s]
                if (day - when).days <= cfg["reentry_days"]:
                    if closes.shift(1).loc[day, s] < px * (1 + cfg["reentry_rebound_pct"] / 100):
                        continue  # still waiting for the rebound
                elif sc[s] <= cfg["news_block_buy_score"]:
                    continue
                sold.pop(s, None)
            elif use_news and sc[s] <= cfg["news_block_buy_score"]:
                continue
            buy = min(size, cash)
            if buy < 1:
                break
            held[s] = buy / o[s]; cash -= buy; trades += 1
        curve.append(cash + sum(q * closes.loc[day, k] for k, q in held.items()))
    eq = pd.Series(curve, index=[d for d in closes.index if d >= score.index.min()])
    return eq, trades


def main():
    cfg = load_config()
    uni = cfg["universe"]
    print(f"Downloading/caching ~{DAYS} days of Alpaca headlines for {len(uni)} stocks...")
    news = fetch_news(uni)
    print(f"{len(news):,} focused headlines, {news.time.min():%Y-%m-%d} to {news.time.max():%Y-%m-%d}; "
          f"{(news.score < 0).sum():,} scored negative, {(news.score > 0).sum():,} positive")

    px = yf.download(uni, start=(datetime.now() - timedelta(days=DAYS + 220)).strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False, threads=False)
    opens, closes = px["Open"][uni], px["Close"][uni]
    window = closes.index[closes.index >= news.day.min()]
    score = daily_scores(news, window, uni)

    print("\nEVENT STUDY (buy at the open after the signal; what happened next)")
    rows = event_study(score, opens.loc[window], closes.loc[window], cfg["news_sell_score"], f"score <= {cfg['news_sell_score']} (sell)")
    rows += event_study(score, opens.loc[window], closes.loc[window], cfg["news_block_buy_score"], f"score <= {cfg['news_block_buy_score']} (block buy)")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nREPLAY over the news window (top-10 momentum)")
    for use in (False, True):
        eq, tr = replay(opens, closes, score, cfg, use)
        ret = eq.iloc[-1] / eq.iloc[0] - 1
        dd = (eq / eq.cummax() - 1).min()
        print(f"  {'with news rules   ' if use else 'without news rules'}: return {ret*100:+.2f}%, worst drop {dd*100:.1f}%, trades {tr}")

    worst = news.sort_values("score").head(8)
    print("\nSample of headlines scored most negative:")
    for _, r in worst.iterrows():
        print(f"  {r.time:%Y-%m-%d} {r.symbol:5s} {r.score:+d} {r.headline[:90]}")


if __name__ == "__main__":
    main()
