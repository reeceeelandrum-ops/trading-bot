"""Moving-average crossover trading bot for an Alpaca PAPER account.

Run once a day (e.g. after the market opens):
    python bot.py            # checks for a crossover and trades if one happened
    python bot.py --dry-run  # shows the signal without placing an order
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


def detect_crossover(close: pd.Series, short: int, long: int) -> tuple[str | None, pd.DataFrame]:
    """Return 'buy', 'sell', or None, comparing the last two consecutive days."""
    df = pd.DataFrame({"close": close})
    df["sma_short"] = close.rolling(short).mean()
    df["sma_long"] = close.rolling(long).mean()
    df = df.dropna()
    if len(df) < 2:
        sys.exit("Not enough data to compare two days of moving averages")

    prev, today = df.iloc[-2], df.iloc[-1]
    if prev.sma_short <= prev.sma_long and today.sma_short > today.sma_long:
        return "buy", df
    if prev.sma_short >= prev.sma_long and today.sma_short < today.sma_long:
        return "sell", df
    return None, df


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

    close = fetch_prices(symbol, cfg["lookback_days"])
    signal, df = detect_crossover(close, short, long)

    last = df.iloc[-1]
    print(f"{symbol} on {df.index[-1].date()}: close={last.close:.2f}  "
          f"SMA{short}={last.sma_short:.2f}  SMA{long}={last.sma_long:.2f}")

    if signal is None:
        print("No crossover between the last two days. No trade.")
        return

    print(f"Crossover detected: {signal.upper()} signal")
    if args.dry_run:
        print("Dry run: no order placed.")
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

    qty = cfg["trade_size"]
    if signal == "sell":
        owned = held_qty(client, symbol)
        if owned <= 0:
            print(f"No {symbol} shares held, so nothing to sell (bot doesn't short).")
            return
        qty = min(qty, owned)

    order = place_order(client, symbol, signal, qty)
    print(f"Order submitted: {signal} {qty} {symbol}  id={order.id}  status={order.status}")


if __name__ == "__main__":
    main()
