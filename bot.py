"""Rule-based multi-stock trend bot for an Alpaca PAPER account.

Every run, for each stock in config.json's "universe":
  gap = how far the short moving average is above (+) or below (-) the long one, in %.

Rules (applied the same way every day, no judgment calls; see signals.py):
  SELL  a stock the bot holds once its gap falls below -buffer_pct (clear downtrend),
        or its recent news score drops to news_sell_score or lower.
  BUY   stocks with a gap above +buffer_pct (clear uptrend), strongest gap first, that
          - don't report earnings within earnings_blackout_days,
          - are profitable (if require_profitable) with revenue growth >= min_revenue_growth_pct,
          - have a news score above news_block_buy_score,
        until the bot holds max_positions stocks or cash would drop below min_cash.
  Between -buffer_pct and +buffer_pct nothing changes (filters out whipsaws).
Positions in stocks outside the universe are never touched.

Run once a day during market hours:
    python bot.py            # checks every stock and trades if needed
    python bot.py --dry-run  # shows the plan without connecting or trading
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from signals import NewsScorer, company_check, earnings_check

HERE = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")


def load_config() -> dict:
    with open(HERE / "config.json") as f:
        cfg = json.load(f)
    if cfg["short_window"] >= cfg["long_window"]:
        sys.exit("config error: short_window must be smaller than long_window")
    if cfg["lookback_days"] < cfg["long_window"] + 1:
        sys.exit("config error: lookback_days must be greater than long_window")
    cfg["universe"] = [s.upper() for s in cfg["universe"]]
    return cfg


def fetch_closes(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """Daily closing prices (one column per symbol) for the last `lookback_days` trading days."""
    start = date.today() - timedelta(days=lookback_days * 2)  # covers weekends/holidays
    data = yf.download(symbols, start=start.isoformat(), interval="1d",
                       auto_adjust=True, progress=False)
    if data.empty:
        sys.exit("No price data returned")
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    return closes.tail(lookback_days)


def trend_table(closes: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """One row per symbol: latest close, both averages, gap %, and the date of the latest bar."""
    rows = {}
    for sym in closes.columns:
        c = closes[sym].dropna()
        if len(c) < long:
            print(f"  {sym}: not enough price history, skipped")
            continue
        s, l = c.rolling(short).mean().iloc[-1], c.rolling(long).mean().iloc[-1]
        rows[sym] = {"close": c.iloc[-1], "sma_short": s, "sma_long": l,
                     "gap_pct": (s - l) / l * 100, "bar_date": c.index[-1].date()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("gap_pct", ascending=False)


def plan_trades(table: pd.DataFrame, held: dict[str, float], cash: float, cfg: dict,
                buy_check=lambda sym: (True, ""), news_score=lambda sym: None) -> list[tuple]:
    """Pure decision logic. Returns [(side, symbol, qty, reason), ...].

    buy_check(sym) -> (ok, note): earnings + company checks for a buy candidate.
    news_score(sym) -> int | None: summed headline score (None = no news data).
    """
    buffer, size = cfg["buffer_pct"], cfg["trade_size"]
    trades = []

    # 1) Sells: held stocks in a clear downtrend, or with very bad recent news.
    holding = {s for s, q in held.items() if q > 0 and s in table.index}
    for sym in sorted(holding):
        gap = table.at[sym, "gap_pct"]
        reason = None
        if gap < -buffer:
            reason = f"downtrend, gap {gap:+.2f}%"
        else:
            score = news_score(sym)
            if score is not None and score <= cfg["news_sell_score"]:
                reason = f"bad news, score {score:+d}"
        if reason:
            trades.append(("sell", sym, min(held[sym], size), reason))
            holding.discard(sym)

    # 2) Buys: strongest clear uptrends first that pass the earnings, company, and
    #    news checks, within the position and cash limits.
    for sym, row in table.iterrows():  # already sorted strongest first
        if len(holding) >= cfg["max_positions"]:
            break
        if sym in holding or row.gap_pct <= buffer:
            continue
        ok, note = buy_check(sym)
        if not ok:
            print(f"  {sym}: skipped, {note}")
            continue
        score = news_score(sym)
        if score is not None and score <= cfg["news_block_buy_score"]:
            print(f"  {sym}: skipped, negative news (score {score:+d})")
            continue
        cost = row.close * size
        if cash - cost < cfg["min_cash"]:
            print(f"  {sym}: skipped, buying would take cash below ${cfg['min_cash']:,}")
            continue
        news = "no news data" if score is None else f"news {score:+d}"
        trades.append(("buy", sym, size, f"uptrend, gap {row.gap_pct:+.2f}%; {note}; {news}"))
        holding.add(sym)
        cash -= cost
    return trades


def get_trading_client():
    from alpaca.trading.client import TradingClient

    load_dotenv(HERE / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret or key.startswith("your_"):
        sys.exit("Add your Alpaca paper API keys to the .env file first")
    return TradingClient(key, secret, paper=True)  # paper=True: never touches real money


def held_positions(client) -> dict[str, float]:
    return {p.symbol: float(p.qty) for p in client.get_all_positions()}


def symbols_traded_today(client, now: datetime) -> set[str]:
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    midnight = now.astimezone(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=midnight, limit=500))
    return {o.symbol for o in orders if o.status not in ("canceled", "rejected", "expired")}


def place_order(client, symbol: str, side: str, qty: float):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    return client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="don't connect or place orders")
    args = parser.parse_args()
    cfg = load_config()
    load_dotenv(HERE / ".env")  # keys are also used for the read-only news check in dry runs

    client, held, cash, today = None, {}, float("inf"), None
    if not args.dry_run:
        client = get_trading_client()
        account, clock = client.get_account(), client.get_clock()
        cash = float(account.cash)
        print(f"Connected to paper account {account.account_number} "
              f"(cash ${cash:,.2f}). Market open: {clock.is_open}")
        if not clock.is_open:
            print("Market is closed (weekend, holiday, or outside hours). No trades.")
            return
        held = held_positions(client)
        today = clock.timestamp.astimezone(NY).date()

    closes = fetch_closes(cfg["universe"], cfg["lookback_days"])
    table = trend_table(closes, cfg["short_window"], cfg["long_window"])

    print(f"\nTrend ranking (SMA{cfg['short_window']} vs SMA{cfg['long_window']}, "
          f"buffer ±{cfg['buffer_pct']}%):")
    for sym, r in table.iterrows():
        mark = "held" if held.get(sym, 0) > 0 else ""
        print(f"  {sym:6s} close {r.close:9.2f}  gap {r.gap_pct:+6.2f}%  {mark}")

    if today is not None:
        stale = [s for s, d in table.bar_date.items() if d != today]
        if stale:
            print(f"Skipping stocks without today's price data: {', '.join(stale)}")
            table = table.drop(stale)

    news = NewsScorer(cfg["news_lookback_days"])
    if news.client is None:
        print("No API keys available, so the news check is off for this run.")

    def buy_check(sym):
        ok, e_note = earnings_check(sym, cfg["earnings_blackout_days"])
        if not ok:
            return False, e_note
        ok, c_note = company_check(sym, cfg["require_profitable"], cfg["min_revenue_growth_pct"])
        return ok, c_note if not ok else f"{c_note}, {e_note}"

    print("\nChecks:")
    trades = plan_trades(table, held, cash, cfg, buy_check, lambda s: news.score(s)[0])

    # Show the headlines behind every news score that was used.
    for sym, (score, lines) in news.cache.items():
        if lines:
            print(f"  {sym} news {score:+d}:")
            for line in lines[:5]:
                print(f"     {line[:110]}")
    if not trades:
        print("\nNo trades needed today.")
        return

    print("\nPlan:")
    for side, sym, qty, reason in trades:
        print(f"  {side.upper():4s} {qty:g} {sym:6s} ({reason})")
    if args.dry_run:
        print("Dry run (assumes nothing held): no orders placed.")
        return

    already = symbols_traded_today(client, clock.timestamp)
    for side, sym, qty, reason in trades:
        if sym in already:
            print(f"  {sym}: already traded today, skipped")
            continue
        try:
            o = place_order(client, sym, side, qty)
            print(f"  Submitted {side} {qty:g} {sym}: id={o.id} status={o.status}")
        except Exception as e:  # one failed order shouldn't stop the rest
            print(f"  FAILED {side} {qty:g} {sym}: {e}")


if __name__ == "__main__":
    main()
