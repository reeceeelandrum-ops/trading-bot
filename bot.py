"""Rule-based trading bot for an Alpaca PAPER account. Runs every 15 minutes.

STOCKS (config "universe", during regular market hours)
  Monthly momentum: rank the universe by return over the last momentum_lookback_days
  trading days, measured at the end of the previous month. The top max_positions are
  this month's targets, fixed for the whole month.
  BUY   targets not held, strongest first, that pass the earnings, company and news
        checks (see signals.py), while cash stays above min_cash.
  SELL  held universe stocks that are no longer targets.

CRYPTO (config "crypto", 24/7)
  Trend: hold crypto.dollars of each coin while its short daily moving average is more
  than buffer_pct above the long one; sell when it's more than buffer_pct below.

BOTH, every run
  NEWS SELL   sell a holding whose recent news score is news_sell_score or lower.
  REBOUND     within reentry_days of a news sell, buy it back if the price rises
              reentry_rebound_pct above the sale price, unless its rules now say no
              (stock no longer a target, or crypto in a clear downtrend).
  At most one buy and one sell per symbol per day. Positions outside the bot's lists
  are never touched.

    python bot.py            # check everything and trade if needed
    python bot.py --dry-run  # show the plan without connecting or trading
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from signals import NewsScorer, company_check, earnings_check

HERE = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")
PUBLIC_LOG = bool(os.getenv("GITHUB_ACTIONS"))  # hide balances in public run logs


def load_config() -> dict:
    with open(HERE / "config.json") as f:
        cfg = json.load(f)
    if cfg["short_window"] >= cfg["long_window"]:
        sys.exit("config error: short_window must be smaller than long_window")
    cfg["universe"] = [s.upper() for s in cfg["universe"]]
    cfg.setdefault("crypto", {"symbols": [], "dollars": 0})
    return cfg


def norm(sym: str) -> str:
    """Alpaca writes crypto as BTC/USD in orders and BTCUSD in positions."""
    return sym.replace("/", "")


# ---------- prices ----------

def fetch_closes(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """Daily closes (one column per symbol) for about the last `lookback_days` trading days."""
    start = date.today() - timedelta(days=int(lookback_days * 1.6) + 10)
    # threads=False: parallel downloads can clash on yfinance's cache ("database is locked").
    data = yf.download(symbols, start=start.isoformat(), interval="1d", auto_adjust=True,
                       progress=False, threads=False)
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
            continue
        s, l = c.rolling(short).mean().iloc[-1], c.rolling(long).mean().iloc[-1]
        rows[sym] = {"close": c.iloc[-1], "sma_short": s, "sma_long": l,
                     "gap_pct": (s - l) / l * 100, "bar_date": c.index[-1].date()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("gap_pct", ascending=False)


def momentum_ranking(closes: pd.DataFrame, lookback: int, as_of: date) -> pd.Series:
    """Return % over `lookback` trading days, ending on the last trading day before the
    month that `as_of` is in. Fixed for the whole month."""
    idx = closes.index.tz_localize(None) if closes.index.tz is not None else closes.index
    hist = closes[idx < pd.Timestamp(as_of.replace(day=1))]
    if len(hist) <= lookback:
        sys.exit(f"Not enough history for {lookback}-day momentum")
    end, start = hist.iloc[-1], hist.iloc[-1 - lookback]
    return ((end / start - 1) * 100).dropna().sort_values(ascending=False)


def crypto_daily_closes(symbols: list[str], days: int) -> pd.DataFrame:
    """Completed daily (UTC) closes for crypto pairs, one column per symbol."""
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start = datetime.now(timezone.utc) - timedelta(days=days + 5)
    df = CryptoHistoricalDataClient().get_crypto_bars(
        CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start)).df
    closes = df["close"].unstack(level=0)
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")
    return closes[closes.index < today_utc]  # drop today's unfinished bar


def latest_prices(stocks: list[str], cryptos: list[str]) -> dict[str, float]:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import CryptoLatestTradeRequest, StockLatestTradeRequest

    out = {}
    try:
        if stocks:
            s = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
            out.update({k: float(v.price) for k, v in s.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=stocks, feed=DataFeed.IEX)).items()})
        if cryptos:
            out.update({k: float(v.price) for k, v in CryptoHistoricalDataClient().get_crypto_latest_trade(
                CryptoLatestTradeRequest(symbol_or_symbols=cryptos)).items()})
    except Exception as e:
        print(f"  could not read latest prices: {e}")
    return out


# ---------- decisions (pure functions, no network) ----------

def stock_dollars(cfg: dict, equity: float) -> float:
    """Dollars per stock. "auto" splits what's left after the cash floor and the crypto
    budget equally across max_positions; a number is used as is."""
    v = cfg["stock_dollars"]
    if v != "auto":
        return float(v)
    crypto = cfg["crypto"]["dollars"] * len(cfg["crypto"]["symbols"])
    return max(0.0, (equity - cfg["min_cash"] - crypto) / cfg["max_positions"])


def plan_stocks(targets: list[str], universe: set[str], held: dict[str, float], prices: dict[str, float],
                cash: float, cfg: dict, buy_check, news_score, news_sold: dict[str, float],
                equity: float = 100_000.0) -> list[tuple]:
    trades = []
    dollars = stock_dollars(cfg, equity)

    def shares_for(price, have=0.0):
        """Whole shares to bring a position up to about `dollars` worth."""
        return max(0, int(dollars // price) - int(have))

    bot_held = {s for s, q in held.items() if q > 0 and s in universe}

    for sym in sorted(bot_held):
        if sym not in targets:
            trades.append(("sell", sym, held[sym], "no longer a top-momentum stock this month"))
            continue
        score = news_score(sym)
        if score is not None and score <= cfg["news_sell_score"]:
            trades.append(("sell", sym, held[sym], f"bad news, score {score:+d}"))
            continue
        price = prices.get(sym)
        if price and held[sym] * price > dollars * 1.25:  # grown well past its share: trim
            extra = int(held[sym]) - int(dollars // price)
            if extra > 0:
                trades.append(("sell", sym, extra, f"trim to target size (${held[sym]*price:,.0f} vs ${dollars:,.0f})"))
    selling = {t[1] for t in trades if not t[3].startswith("trim")}
    # Sales go out before buys in the same run, so count their proceeds as available cash.
    cash += sum(q * prices.get(s, 0.0) for side, s, q, _ in trades if side == "sell")

    for sym in targets:
        if sym in selling:
            continue
        price = prices.get(sym)
        if price is None:
            print(f"  {sym}: skipped, no current price")
            continue
        have = held.get(sym, 0.0) if sym in bot_held else 0.0
        if have and have * price >= dollars * 0.8:
            continue  # already close enough to the target size
        qty = shares_for(price, have)
        if qty == 0:
            if not have:
                print(f"  {sym}: skipped, one share (${price:,.2f}) is more than ${dollars:,} per stock")
            continue
        # Never let a buy take cash below the floor: shrink it to fit.
        room = cash - cfg["min_cash"]
        if room != float("inf"):  # dry runs have no account, so cash is unlimited
            qty = min(qty, int(max(0.0, room) // price))
        if qty == 0:
            print(f"  {sym}: skipped, buying would take cash below ${cfg['min_cash']:,}")
            continue
        top_up = f"top-up from {have:g} shares; " if have else ""
        if sym in news_sold:
            sold_at = news_sold[sym]
            if price >= sold_at * (1 + cfg["reentry_rebound_pct"] / 100):
                trades.append(("buy", sym, qty, f"rebound: {price:.2f} is {price/sold_at*100-100:+.1f}% above news-sale price {sold_at:.2f}"))
                cash -= price * qty
            else:
                print(f"  {sym}: sold on bad news at {sold_at:.2f}; waiting for a {cfg['reentry_rebound_pct']}% rebound (now {price:.2f})")
            continue
        ok, note = buy_check(sym)
        if not ok:
            print(f"  {sym}: skipped, {note}")
            continue
        score = news_score(sym)
        if score is not None and score <= cfg["news_block_buy_score"]:
            print(f"  {sym}: skipped, negative news (score {score:+d})")
            continue
        trades.append(("buy", sym, qty, f"{top_up}top-momentum target, ~${qty*price:,.0f}; {note}; news {'n/a' if score is None else f'{score:+d}'}"))
        cash -= price * qty
    return trades


def plan_crypto(table: pd.DataFrame, held: dict[str, float], prices: dict[str, float], cash: float,
                cfg: dict, news_score, news_sold: dict[str, float]) -> list[tuple]:
    trades, buf, dollars = [], cfg["buffer_pct"], cfg["crypto"]["dollars"]
    for sym, row in table.iterrows():
        qty = held.get(norm(sym), 0)
        gap, score = row.gap_pct, news_score(sym)
        if qty > 0:
            if gap < -buf:
                trades.append(("sell", sym, qty, f"downtrend, gap {gap:+.2f}%"))
            elif score is not None and score <= cfg["news_sell_score"]:
                trades.append(("sell", sym, qty, f"bad news, score {score:+d}"))
            continue
        price = prices.get(sym)
        wants = (gap > buf) or (norm(sym) in news_sold and gap >= -buf)
        if wants and cash - dollars < cfg["min_cash"]:
            print(f"  {sym}: skipped, buying would take cash below ${cfg['min_cash']:,}")
            continue
        if norm(sym) in news_sold:
            sold_at = news_sold[norm(sym)]
            if gap >= -buf and price and price >= sold_at * (1 + cfg["reentry_rebound_pct"] / 100):
                trades.append(("buy", sym, dollars, f"rebound: {price/sold_at*100-100:+.1f}% above news-sale price"))
                cash -= dollars
            continue
        if gap > buf:
            if score is not None and score <= cfg["news_block_buy_score"]:
                print(f"  {sym}: skipped, negative news (score {score:+d})")
                continue
            trades.append(("buy", sym, dollars, f"uptrend, gap {gap:+.2f}%"))
            cash -= dollars
    return trades


# ---------- account ----------

def get_trading_client():
    from alpaca.trading.client import TradingClient

    load_dotenv(HERE / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret or key.startswith("your_"):
        sys.exit("Add your Alpaca paper API keys to the .env file first")
    return TradingClient(key, secret, paper=True)  # paper=True: never touches real money


def held_positions(client) -> dict[str, float]:
    return {norm(p.symbol): float(p.qty) for p in client.get_all_positions()}


def recent_orders(client, since: datetime):
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    return client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=500))


def place_order(client, symbol: str, side: str, amount: float, tag: str, crypto: bool = False):
    """Stocks: `amount` is shares. Crypto: buys take dollars, sells take quantity."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    kw = {"symbol": symbol, "side": OrderSide.BUY if side == "buy" else OrderSide.SELL,
          "time_in_force": TimeInForce.GTC if crypto else TimeInForce.DAY,
          "client_order_id": f"{tag}-{norm(symbol)}-{int(time.time())}"}
    if crypto and side == "buy":
        kw["notional"] = round(amount, 2)
    else:
        kw["qty"] = amount
    return client.submit_order(MarketOrderRequest(**kw))


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="don't connect or place orders")
    args = parser.parse_args()
    cfg = load_config()
    load_dotenv(HERE / ".env")  # keys are also used for read-only news and prices in dry runs
    now = datetime.now(timezone.utc)
    cryptos = cfg["crypto"]["symbols"]

    client, held, cash, market_open, equity = None, {}, float("inf"), True, 100_000.0
    news_sold, done_today = {}, set()
    if not args.dry_run:
        client = get_trading_client()
        account, clock = client.get_account(), client.get_clock()
        cash, market_open, equity = float(account.cash), clock.is_open, float(account.equity)
        print("Connected to paper account." if PUBLIC_LOG else
              f"Connected to paper account {account.account_number} (cash ${cash:,.2f}).")
        print(f"Stock market open: {market_open}")
        held = held_positions(client)
        midnight = now.astimezone(NY).replace(hour=0, minute=0, second=0, microsecond=0)
        for o in recent_orders(client, now - timedelta(days=cfg["reentry_days"] + 3)):
            st = getattr(o.status, "value", o.status)
            if st in ("canceled", "rejected", "expired"):
                continue
            side = getattr(o.side, "value", o.side)
            if o.submitted_at >= midnight:
                done_today.add((norm(o.symbol), side))
            if (side == "sell" and (o.client_order_id or "").startswith("news-") and o.filled_avg_price
                    and o.filled_at and o.filled_at >= now - timedelta(days=cfg["reentry_days"])):
                news_sold[norm(o.symbol)] = float(o.filled_avg_price)

    news = NewsScorer(cfg["news_lookback_days"])
    trades = []  # (side, symbol, amount, reason, is_crypto)

    # ----- stocks -----
    if market_open:
        closes = fetch_closes(cfg["universe"], cfg["momentum_lookback_days"] + 40)
        today = now.astimezone(NY).date()
        rank = momentum_ranking(closes, cfg["momentum_lookback_days"], today)
        # Targets: the strongest stocks that pass the company check (a lasting test, so a
        # failing stock gives its slot to the next one). Earnings and news are short-lived,
        # so they only delay a buy.
        targets, company_notes = [], {}
        for sym in rank.index:
            if len(targets) >= cfg["max_positions"]:
                break
            ok, note = company_check(sym, cfg["require_profitable"], cfg["min_revenue_growth_pct"])
            if ok:
                targets.append(sym)
                company_notes[sym] = note
            else:
                print(f"  {sym}: left out of targets, {note}")
        print(f"\nMomentum ranking for {today:%B %Y} ({cfg['momentum_lookback_days']} trading days to end of last month):")
        for i, (sym, r) in enumerate(rank.items(), 1):
            tag = "TARGET" if sym in targets else ""
            print(f"  {i:2d}. {sym:6s} {r:+7.2f}%  {tag}{'  held' if held.get(sym, 0) > 0 else ''}")
        need = [s for s in set(targets) | {s for s in held if s in cfg['universe']}]
        prices = latest_prices(need, [])

        def buy_check(sym):
            ok, e_note = earnings_check(sym, cfg["earnings_blackout_days"])
            return ok, e_note if not ok else f"{company_notes.get(sym, '')}, {e_note}"

        print("\nStock checks:")
        for side, sym, amt, why in plan_stocks(targets, set(cfg["universe"]), held, prices, cash, cfg,
                                               buy_check, lambda s: news.score(s)[0], news_sold, equity):
            trades.append((side, sym, amt, why, False))
            cash += (amt if side == "sell" else -amt) * prices.get(sym, 0)
    else:
        print("Stock market closed: skipping stocks this run.")

    # ----- crypto -----
    if cryptos:
        ctable = trend_table(crypto_daily_closes(cryptos, cfg["long_window"] + 10), cfg["short_window"], cfg["long_window"])
        print(f"\nCrypto trend (daily SMA{cfg['short_window']} vs SMA{cfg['long_window']}, buffer ±{cfg['buffer_pct']}%):")
        for sym, r in ctable.iterrows():
            print(f"  {sym:8s} close {r.close:12,.2f}  gap {r.gap_pct:+6.2f}%{'  held' if held.get(norm(sym), 0) > 0 else ''}")
        cprices = latest_prices([], cryptos)
        print("\nCrypto checks:")
        for side, sym, amt, why in plan_crypto(ctable, held, cprices, cash, cfg, lambda s: news.score(s)[0], news_sold):
            trades.append((side, sym, amt, why, True))

    for sym, (score, lines) in news.cache.items():
        if lines:
            print(f"  {sym} news {score:+d}:")
            for line in lines[:5]:
                print(f"     {line[:110]}")

    if not trades:
        print("\nNo trades needed this run.")
        return
    print("\nPlan:")
    for side, sym, amt, why, is_c in trades:
        amount = f"${amt:,.0f} of" if (is_c and side == "buy") else f"{amt:g}"
        print(f"  {side.upper():4s} {amount} {sym} ({why})")
    if args.dry_run:
        print("Dry run (assumes nothing held): no orders placed.")
        return

    trades.sort(key=lambda t: t[0] != "sell")  # sells first, so their cash funds the buys
    for side, sym, amt, why, is_c in trades:
        if (norm(sym), side) in done_today:
            print(f"  {sym}: already placed a {side} today, skipped")
            continue
        tag = "news" if why.startswith("bad news") else "rebound" if why.startswith("rebound") else "bot"
        try:
            o = place_order(client, sym, side, amt, tag, crypto=is_c)
            print(f"  Submitted {side} {sym}: status={getattr(o.status, 'value', o.status)}")
        except Exception as e:  # one failed order shouldn't stop the rest
            print(f"  FAILED {side} {sym}: {e}")


if __name__ == "__main__":
    main()
