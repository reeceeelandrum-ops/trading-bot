"""Moving-average trading bot for an Alpaca PAPER account.

Two modes (set "mode" in config.json):
  crossover  trade only on the day the short average crosses the long one
  trend      hold `trade_size` shares while the short average is more than
             `buffer_pct` % above the long one, none while it's more than
             `buffer_pct` % below, and leave the position alone in between

Run once a day during market hours:
    python bot.py            # checks the signal and trades if needed
    python bot.py --dry-run  # shows the signal without connecting or trading
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

HERE = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")


def load_config() -> dict:
    with open(HERE / "config.json") as f:
        cfg = json.load(f)
    if cfg["short_window"] >= cfg["long_window"]:
        sys.exit("config error: short_window must be smaller than long_window")
    if cfg["lookback_days"] < cfg["long_window"] + 1:
        sys.exit("config error: lookback_days must be greater than long_window")
    return cfg


def fetch_prices(symbol: str, lookback_days: int) -> pd.Series:
    """Daily closing prices for the last `lookback_days` trading days."""
    # Ask for roughly twice as many calendar days to cover weekends/holidays.
    start = date.today() - timedelta(days=lookback_days * 2)
    data = yf.download(symbol, start=start.isoformat(), interval="1d",
                       auto_adjust=True, progress=False)
    if data.empty:
        sys.exit(f"No price data returned for {symbol}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a column per ticker
        close = close[symbol]
    return close.dropna().tail(lookback_days)


def moving_averages(close: pd.Series, short: int, long: int) -> pd.DataFrame:
    df = pd.DataFrame({"close": close})
    df["sma_short"] = close.rolling(short).mean()
    df["sma_long"] = close.rolling(long).mean()
    df = df.dropna()
    if len(df) < 2:
        sys.exit("Not enough data to compare two days of moving averages")
    # How far the short average is above (+) or below (-) the long one, in percent.
    df["gap_pct"] = (df.sma_short - df.sma_long) / df.sma_long * 100
    return df


def detect_crossover(close: pd.Series, short: int, long: int) -> tuple[str | None, pd.DataFrame]:
    """Return 'buy', 'sell', or None, comparing the last two consecutive days."""
    df = moving_averages(close, short, long)
    prev, today = df.iloc[-2], df.iloc[-1]
    if prev.sma_short <= prev.sma_long and today.sma_short > today.sma_long:
        return "buy", df
    if prev.sma_short >= prev.sma_long and today.sma_short < today.sma_long:
        return "sell", df
    return None, df


def decide_trend(gap_pct: float, buffer_pct: float, held: float, trade_size: float):
    """Trend mode: hold `trade_size` shares while the short average is clearly above
    the long one, hold none while it's clearly below, and change nothing while the two
    are within `buffer_pct` of each other (that dead zone filters out whipsaws).

    Returns (side, qty, reason); side is None when no trade is needed.
    """
    if gap_pct > buffer_pct:
        if held < trade_size:
            return "buy", trade_size - held, f"uptrend (gap {gap_pct:+.2f}% > +{buffer_pct}%)"
        return None, 0, f"uptrend (gap {gap_pct:+.2f}%), already holding {held:g}"
    if gap_pct < -buffer_pct:
        if held > 0:
            # Only sell what the bot would have bought, never more.
            return "sell", min(held, trade_size), f"downtrend (gap {gap_pct:+.2f}% < -{buffer_pct}%)"
        return None, 0, f"downtrend (gap {gap_pct:+.2f}%), holding nothing"
    return None, 0, f"inside the ±{buffer_pct}% buffer (gap {gap_pct:+.2f}%): keeping current position"


def get_trading_client():
    from alpaca.trading.client import TradingClient

    load_dotenv(HERE / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret or key.startswith("your_"):
        sys.exit("Add your Alpaca paper API keys to the .env file first")
    return TradingClient(key, secret, paper=True)  # paper=True: never touches real money


def held_qty(client, symbol: str) -> float:
    try:
        return float(client.get_open_position(symbol).qty)
    except Exception:
        return 0.0


def traded_today(client, symbol: str, now: datetime) -> bool:
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    midnight = now.astimezone(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    orders = client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL, symbols=[symbol], after=midnight))
    return any(o.status not in ("canceled", "rejected", "expired") for o in orders)


def place_order(client, symbol: str, side: str, qty: float):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="don't place orders")
    args = parser.parse_args()

    cfg = load_config()
    symbol = cfg["symbol"]
    short, long = cfg["short_window"], cfg["long_window"]

    client = None
    if not args.dry_run:
        # Connect first: this also confirms the API keys work on every run.
        client = get_trading_client()
        account = client.get_account()
        clock = client.get_clock()
        print(f"Connected to paper account {account.account_number} "
              f"(cash ${float(account.cash):,.2f}). Market open: {clock.is_open}")
        if not clock.is_open:
            print("Market is closed (weekend, holiday, or outside hours). No trade.")
            return

    mode = cfg.get("mode", "crossover")
    buffer_pct = float(cfg.get("buffer_pct", 0))
    trade_size = cfg["trade_size"]

    close = fetch_prices(symbol, cfg["lookback_days"])
    signal, df = detect_crossover(close, short, long)

    last = df.iloc[-1]
    print(f"{symbol} on {df.index[-1].date()}: close={last.close:.2f}  "
          f"SMA{short}={last.sma_short:.2f}  SMA{long}={last.sma_long:.2f}  "
          f"gap={last.gap_pct:+.2f}%  mode={mode}")

    if mode == "trend":
        held = 0.0 if args.dry_run else held_qty(client, symbol)
        signal, qty, reason = decide_trend(last.gap_pct, buffer_pct, held, trade_size)
        print(f"Trend check: {reason}" + (" (dry run assumes 0 shares held)" if args.dry_run else ""))
        if signal is None:
            print("No trade.")
            return
    else:
        if signal is None:
            print("No crossover between the last two days. No trade.")
            return
        print(f"Crossover detected: {signal.upper()} signal")
        qty = trade_size

    if args.dry_run:
        print(f"Dry run: would {signal} {qty:g} {symbol}. No order placed.")
        return

    # Only act on a signal from today's price bar, so a stale signal is never re-traded.
    market_today = clock.timestamp.astimezone(NY).date()
    if df.index[-1].date() != market_today:
        print(f"Latest price data is from {df.index[-1].date()}, not today ({market_today}). No trade.")
        return

    # Don't trade twice in one day (e.g. if the workflow is run again manually).
    if traded_today(client, symbol, clock.timestamp):
        print(f"Already placed a {symbol} order today. No new trade.")
        return

    if signal == "sell" and mode != "trend":
        owned = held_qty(client, symbol)
        if owned <= 0:
            print(f"No {symbol} shares held, so nothing to sell (bot doesn't short).")
            return
        qty = min(qty, owned)

    order = place_order(client, symbol, signal, qty)
    print(f"Order submitted: {signal} {qty} {symbol}  id={order.id}  status={order.status}")


if __name__ == "__main__":
    main()
