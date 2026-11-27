"""Collect a snapshot of the paper account and the bot's view for the dashboard.

Writes one JSON file per dataset into dashboard_data/, each as
{"rows": [...], "meta": [...]} with this script's own source in meta,
so every number on the dashboard can be traced back to how it was made.

    .venv/bin/python dashboard_data.py
"""

import json
import math
import subprocess
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

from bot import (HERE, NY, crypto_daily_closes, fetch_closes, get_trading_client, load_config,
                 momentum_ranking, norm, rebalance_schedule, trend_table)
from signals import company_check

OUT = HERE / "dashboard_data"
SOURCE = Path(__file__).read_text()
NOW = datetime.now(timezone.utc)


def clean(v):
    """JSON-safe values: no NaN/inf, enums as plain text, datetimes as ISO."""
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 4)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "value"):  # enums
        return v.value
    return v


def write(name, rows, note):
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    meta = [
        {"label": "Snapshot taken", "value": NOW.strftime("%Y-%m-%d %H:%M UTC")},
        {"label": "What it holds", "value": note},
        {"label": "dashboard_data.py", "value": SOURCE, "type": "code", "lang": "python"},
    ]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=1))
    print(f"{name}: {len(rows)} rows")


def f(x):
    return None if x is None else float(x)


def main():
    OUT.mkdir(exist_ok=True)
    cfg = load_config()
    client = get_trading_client()

    # ---- account ----
    a = client.get_account()
    clock = client.get_clock()
    hist = client.get_portfolio_history(GetPortfolioHistoryRequest(period="3M", timeframe="1D"))
    start_value = f(hist.base_value) or 100000.0
    equity, last_equity = f(a.equity), f(a.last_equity)
    positions = client.get_all_positions()
    invested = sum(f(p.market_value) for p in positions)
    write("account", [{
        "account": a.account_number,
        "equity": equity,
        "cash": f(a.cash),
        "buying_power": f(a.buying_power),
        "invested": invested,
        "starting_value": start_value,
        "total_pl": equity - start_value,
        "total_pl_pct": (equity - start_value) / start_value * 100,
        "day_pl": equity - last_equity,
        "day_pl_pct": (equity - last_equity) / last_equity * 100 if last_equity else 0,
        "market_open": clock.is_open,
        "next_open": clock.next_open,
        "next_close": clock.next_close,
        "snapshot_at": NOW,
    }], "Alpaca paper account balances; total P/L is equity minus the account's starting value.")

    # ---- positions ----
    write("positions", [{
        "symbol": p.symbol,
        "qty": f(p.qty),
        "avg_cost": f(p.avg_entry_price),
        "price": f(p.current_price),
        "market_value": f(p.market_value),
        "cost_basis": f(p.cost_basis),
        "unrealized_pl": f(p.unrealized_pl),
        "unrealized_pl_pct": f(p.unrealized_plpc) * 100,
        "today_pl": f(p.unrealized_intraday_pl),
        "managed_by_bot": p.symbol in cfg["universe"] or p.symbol in {norm(s) for s in cfg["crypto"]["symbols"]},
    } for p in positions], "Open positions from Alpaca, valued at the latest price.")

    # ---- orders ----
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, nested=False))
    order_rows = [{
        "id": str(o.id),
        "submitted_at": o.submitted_at,
        "date": o.submitted_at.astimezone(NY).strftime("%Y-%m-%d"),
        "symbol": o.symbol,
        "side": o.side,
        "qty": f(o.qty),
        "type": o.type,
        "status": o.status,
        "filled_qty": f(o.filled_qty),
        "filled_price": f(o.filled_avg_price),
        "filled_at": o.filled_at,
        "value": (f(o.filled_qty) or 0) * (f(o.filled_avg_price) or 0),
    } for o in orders]
    write("orders", order_rows, "Every order on the account, newest first, including cancelled ones.")

    # ---- closed trades (FIFO match of filled buys and sells) ----
    lots, closed = defaultdict(deque), []
    for o in sorted((r for r in order_rows if r["filled_qty"]), key=lambda r: r["filled_at"]):
        if o["side"] == "buy":
            lots[o["symbol"]].append([o["filled_qty"], o["filled_price"], o["filled_at"]])
            continue
        left = o["filled_qty"]
        while left > 0 and lots[o["symbol"]]:
            lot = lots[o["symbol"]][0]
            q = min(left, lot[0])
            closed.append({
                "symbol": o["symbol"], "qty": q,
                "bought_at": lot[2], "buy_price": lot[1],
                "sold_at": o["filled_at"], "sell_price": o["filled_price"],
                "pl": (o["filled_price"] - lot[1]) * q,
                "pl_pct": (o["filled_price"] / lot[1] - 1) * 100,
                "win": o["filled_price"] > lot[1],
            })
            lot[0] -= q
            left -= q
            if lot[0] <= 0:
                lots[o["symbol"]].popleft()
    write("closed_trades", closed, "Completed round trips: each sell matched to the oldest unsold buy (first in, first out).")

    # ---- daily equity / P&L ----
    eq_rows = []
    for ts, eq, pl, plpct in zip(hist.timestamp, hist.equity, hist.profit_loss, hist.profit_loss_pct):
        if eq is None or eq == 0:
            continue
        d = datetime.fromtimestamp(ts, NY).strftime("%Y-%m-%d")
        eq_rows.append({"date": d, "equity": eq, "day_pl": pl, "day_pl_pct": (plpct or 0) * 100})
    # Add today's live figure if the history doesn't have it yet and trading has started today.
    now_ny = NOW.astimezone(NY)
    today = now_ny.strftime("%Y-%m-%d")
    traded_today = now_ny.weekday() < 5 and (now_ny.hour, now_ny.minute) >= (9, 30)
    if traded_today and (not eq_rows or eq_rows[-1]["date"] != today):
        eq_rows.append({"date": today, "equity": equity, "day_pl": equity - last_equity,
                        "day_pl_pct": (equity - last_equity) / last_equity * 100 if last_equity else 0})
    trade_days = defaultdict(lambda: {"buys": 0, "sells": 0})
    for o in order_rows:
        if o["filled_qty"]:
            trade_days[o["date"]]["buys" if o["side"] == "buy" else "sells"] += 1
    for r in eq_rows:
        r.update(trade_days.get(r["date"], {"buys": 0, "sells": 0}))
    write("daily", eq_rows, "End-of-day account value and that day's profit or loss, with filled buys and sells per day.")

    # ---- the bot's view: stock momentum ranking (same rules as bot.py) ----
    as_of, _window = rebalance_schedule(cfg, NOW.astimezone(NY).date())
    closes = fetch_closes(cfg["universe"], cfg["momentum_lookback_days"] + 40 + 21 * int(cfg["rebalance_months"]))
    mom = momentum_ranking(closes, cfg["momentum_lookback_days"], as_of)
    trend = trend_table(closes, cfg["short_window"], cfg["long_window"])
    held = {norm(p.symbol) for p in positions}
    targets, left_out = [], {}
    for sym in mom.index:
        if len(targets) >= cfg["max_positions"]:
            break
        ok, note = company_check(sym, cfg["require_profitable"], cfg["min_revenue_growth_pct"])
        (targets.append(sym) if ok else left_out.__setitem__(sym, note))
    rank = []
    for i, (sym, m) in enumerate(mom.items(), 1):
        status = ("Held" if sym in held else "Target" if sym in targets
                  else "Left out" if sym in left_out else "Not a target")
        if sym in held and sym not in targets:
            status = "Held, leaving"
        t = trend.loc[sym] if sym in trend.index else None
        rank.append({"rank": i, "symbol": sym, "kind": "stock", "momentum_pct": m, "status": status,
                     "note": left_out.get(sym, ""), "close": closes[sym].dropna().iloc[-1],
                     "gap_pct": None if t is None else t.gap_pct})
    write("watchlist", rank, f"The bot's {len(rank)} stocks ranked by return over the {cfg['momentum_lookback_days']} trading days "
          f"to the end of last month; the top {cfg['max_positions']} that pass the company check are this month's targets.")

    # ---- crypto trend ----
    crypto_rows, cc = [], None
    if cfg["crypto"]["symbols"]:
        cc = crypto_daily_closes(cfg["crypto"]["symbols"], 70)
        ct = trend_table(cc, cfg["short_window"], cfg["long_window"])
        b = cfg["buffer_pct"]
        for sym, r in ct.iterrows():
            status = ("Held" if norm(sym) in held else "Uptrend" if r.gap_pct > b
                      else "Downtrend" if r.gap_pct < -b else "Neutral")
            crypto_rows.append({"symbol": sym, "close": r.close, "sma_short": r.sma_short, "sma_long": r.sma_long,
                                "gap_pct": r.gap_pct, "status": status, "dollars": cfg["crypto"]["dollars"]})
    write("crypto", crypto_rows, f"Crypto daily trend: {cfg['short_window']}-day vs {cfg['long_window']}-day average; "
          f"held while more than {cfg['buffer_pct']}% above.")

    price_rows = []
    frames = [closes] + ([cc] if cc is not None else [])
    for frame in frames:
        for sym in frame.columns:
            c = frame[sym].dropna()
            s, l = c.rolling(cfg["short_window"]).mean(), c.rolling(cfg["long_window"]).mean()
            for d in c.index[-60:]:
                price_rows.append({"symbol": sym, "date": d.strftime("%Y-%m-%d"), "close": c[d],
                                   "sma_short": s[d], "sma_long": l[d]})
    write("prices", price_rows, "Last 60 days of closing prices with both moving averages, for each stock and coin.")

    # ---- strategy settings ----
    labels = {
        "max_positions": "Most stocks held at once", "stock_dollars": "Dollars per stock",
        "min_cash": "Cash floor ($)", "short_window": "Short average (days)",
        "long_window": "Long average (days)", "buffer_pct": "Trend buffer (%)",
        "earnings_blackout_days": "No buys this many days before earnings",
        "require_profitable": "Only buy profitable companies",
        "min_revenue_growth_pct": "Minimum sales growth (%)",
        "news_lookback_days": "News window (days)",
        "news_block_buy_score": "Skip buys at news score (blank = off)",
        "news_sell_score": "Sell holding at news score",
        "momentum_lookback_days": "Momentum window (trading days)",
        "reentry_days": "Watch for rebound after news sale (days)",
        "reentry_rebound_pct": "Buy back after rebound of (%)",
    }
    rows = [{"setting": labels[k], "key": k, "value": str(cfg[k])} for k in labels if k in cfg]
    rows += [{"setting": "Crypto traded", "key": "crypto_symbols", "value": ", ".join(cfg["crypto"]["symbols"])},
             {"setting": "Dollars per coin", "key": "crypto_dollars", "value": str(cfg["crypto"]["dollars"])}]
    write("settings", rows, "The bot's rules from config.json.")

    # ---- bot runs on GitHub ----
    gh = HERE / ".tools" / "gh"
    try:
        out = subprocess.run([str(gh), "run", "list", "--workflow", "trading-bot.yml", "-L", "20", "--json",
                              "databaseId,createdAt,event,conclusion,status,url"],
                             cwd=HERE, capture_output=True, text=True, timeout=60, check=True).stdout
        runs = [{"run_id": str(r["databaseId"]), "started_at": r["createdAt"],
                 "trigger": "Scheduled" if r["event"] == "schedule" else "Manual",
                 "result": r["conclusion"] or r["status"], "url": r["url"]} for r in json.loads(out)]
    except Exception as e:
        print("could not read GitHub runs:", e)
        runs = []
    write("bot_runs", runs, "The bot's last 20 runs on GitHub Actions.")


if __name__ == "__main__":
    main()
