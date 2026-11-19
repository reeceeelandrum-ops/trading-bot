"""Write the trading snapshot into the Obsidian vault as linked notes.

Reads the JSON files that dashboard_data.py writes (run that first), then
creates or updates notes in the vault. Only the text between
<!-- trading-bot:start --> and <!-- trading-bot:end --> and the bot's own
properties are rewritten; anything else in a note is kept as you wrote it.

    .venv/bin/python dashboard_data.py && .venv/bin/python obsidian_sync.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "dashboard_data"
VAULT = Path(os.environ.get("OBSIDIAN_VAULT", Path.home() /
             "Library/Mobile Documents/iCloud~md~obsidian/Documents/Trading bot"))
DASHBOARD_URL = "https://claude.ai/artifact/WojFw5RoFR2kmfpK2HCHcV"
REPO_URL = "https://github.com/reeceeelandrum-ops/trading-bot"

START, END = "<!-- trading-bot:start -->", "<!-- trading-bot:end -->"
OPEN_STATES = {"new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding"}


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())["rows"]


def usd(v, d=2):
    return "—" if v is None else f"${float(v):,.{d}f}"


def signed(v):
    if v is None:
        return "—"
    v = float(v)
    return ("+" if v > 0 else "−" if v < 0 else "") + f"${abs(v):,.2f}"


def pct(v):
    if v is None:
        return "—"
    v = float(v)
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.2f}%"


def cell(c):
    # Inside a table, the | in [[note|alias]] must be escaped or it splits the column.
    return re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"[[\1\\|\2]]", str(c))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def yaml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (str, list)):
        return json.dumps(v)
    return v


# ---------- note writing that keeps the user's own text ----------

FM = re.compile(r"\A---\n(.*?)\n---\n?", re.S)


def write_note(path: Path, props: dict, body: str, extra_if_new: str = ""):
    """Update the bot's properties and managed block; keep everything else."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text() if path.exists() else ""

    # Properties: keep the user's own keys, replace the bot's.
    kept, rest = [], old
    m = FM.match(old)
    if m:
        rest = old[m.end():]
        kept = [l for l in m.group(1).splitlines() if l.split(":", 1)[0].strip() not in props]
    fm = "---\n" + "\n".join(kept + [f"{k}: {yaml_value(v)}" for k, v in props.items()]) + "\n---\n"

    block = f"{START}\n{body.strip()}\n{END}"
    if START in rest and END in rest:
        rest = rest[:rest.index(START)] + block + rest[rest.index(END) + len(END):]
    elif rest.strip():
        rest = rest.rstrip() + "\n\n" + block + "\n"
    else:
        rest = block + "\n" + extra_if_new
    path.write_text(fm + rest.lstrip("\n"))


# ---------- notes ----------

def main():
    if not VAULT.exists():
        raise SystemExit(f"Obsidian vault not found: {VAULT}")
    acct = load("account")[0]
    positions = load("positions")
    orders = sorted(load("orders"), key=lambda o: o["submitted_at"] or "", reverse=True)
    closed = load("closed_trades")
    daily = load("daily")
    watch = load("watchlist")
    settings = load("settings")
    runs = load("bot_runs")

    snap = datetime.fromisoformat(acct["snapshot_at"])
    from zoneinfo import ZoneInfo
    ny = snap.astimezone(ZoneInfo("America/New_York"))
    today = ny.strftime("%Y-%m-%d")
    stamp = snap.strftime("%Y-%m-%d %H:%M UTC")

    filled = [o for o in orders if o["filled_qty"]]
    waiting = [o for o in orders if o["status"] in OPEN_STATES]
    realized = sum(float(t["pl"]) for t in closed)
    unrealized = sum(float(p["unrealized_pl"]) for p in positions)
    wins = sum(1 for t in closed if t["win"])
    win_rate = f"{wins / len(closed) * 100:.0f}%" if closed else "— (no completed trades yet)"
    held = {p["symbol"] for p in positions}

    def stock_link(sym):
        return f"[[Stocks/{sym}|{sym}]]"

    # --- Daily journal note ---
    todays = [o for o in orders if o["date"] == today]
    todays_runs = [r for r in runs if (r["started_at"] or "").startswith(today)]
    run_line = ", ".join(f"[{r['trigger']} · {r['result']}]({r['url']})" for r in todays_runs) or "No run recorded yet today."
    day_row = next((d for d in daily if d["date"] == today), None)
    day_pl = day_row["day_pl"] if day_row else acct["day_pl"]
    write_note(VAULT / "Daily" / f"{today}.md", {
        "date": today,
        "account_value": round(acct["equity"], 2),
        "day_pl": round(day_pl, 2),
        "total_pl": round(acct["total_pl"], 2),
        "cash": round(acct["cash"], 2),
        "stocks_held": len(positions),
        "orders_today": len(todays),
        "tags": ["trading/daily"],
    }, f"""
# Trading day · {ny.strftime('%A, %B %-d, %Y')}

> Snapshot {stamp} · [[Home base]] · [Open dashboard]({DASHBOARD_URL})

| Account value | Today | Total since start | Cash |
|---|---|---|---|
| {usd(acct['equity'])} | {signed(day_pl)} | {signed(acct['total_pl'])} ({pct(acct['total_pl_pct'])}) | {usd(acct['cash'])} |

## Orders today
{table(['Side', 'Stock', 'Shares', 'Fill price', 'Status'],
       [[o['side'], stock_link(o['symbol']), f"{o['qty']:g}", usd(o['filled_price']), o['status']] for o in todays])
 if todays else 'No orders today.'}

## Holdings at snapshot
{table(['Stock', 'Shares', 'Paid', 'Now', 'Profit / loss'],
       [[stock_link(p['symbol']), f"{p['qty']:g}", usd(p['avg_cost']), usd(p['price']),
         f"{signed(p['unrealized_pl'])} ({pct(p['unrealized_pl_pct'])})"] for p in positions])
 if positions else 'No stocks held.'}

## Bot run
{run_line}

## Top of the bot's ranking
{', '.join(f"{stock_link(w['symbol'])} {pct(w['gap_pct'])}" for w in watch[:5])}
""", extra_if_new="\n## My notes\n\n")

    # --- Crypto notes ---
    crypto = load("crypto") if (DATA / "crypto.json").exists() else []
    for c in crypto:
        name = c["symbol"].replace("/", "-")
        c_orders = [o for o in orders if o["symbol"].replace("/", "") == c["symbol"].replace("/", "")]
        pos = next((p for p in positions if p["symbol"] == c["symbol"].replace("/", "")), None)
        write_note(VAULT / "Crypto" / f"{name}.md", {
            "symbol": c["symbol"], "status": c["status"], "gap_pct": round(c["gap_pct"], 2),
            "close": round(c["close"], 2), "held": pos is not None, "updated": today, "tags": ["trading/crypto"],
        }, f"""
# {c['symbol']}

**{c['status']}** · trend gap {pct(c['gap_pct'])} · price {usd(c['close'])} · updated {stamp}
20-day average {usd(c['sma_short'])} · 50-day average {usd(c['sma_long'])}. The bot holds {usd(c['dollars'], 0)} of it while the gap is clearly positive.

{f"Holding {pos['qty']:g} worth {usd(pos['market_value'])}: {signed(pos['unrealized_pl'])} ({pct(pos['unrealized_pl_pct'])})." if pos else 'Not held.'}

## Orders
{table(['Date', 'Side', 'Amount', 'Fill price', 'Status'],
       [[f"[[Daily/{o['date']}|{o['date']}]]", o['side'], f"{o['qty']:g}" if o['qty'] else '—', usd(o['filled_price']), o['status']] for o in c_orders])
 if c_orders else 'No orders yet.'}

Back to [[Home base]] · [[Strategy]]
""", extra_if_new="\n## My notes\n\n")

    # --- Stock notes ---
    for w in watch:
        sym = w["symbol"]
        sym_orders = [o for o in orders if o["symbol"] == sym]
        pos = next((p for p in positions if p["symbol"] == sym), None)
        write_note(VAULT / "Stocks" / f"{sym}.md", {
            "symbol": sym,
            "status": w["status"],
            "rank": w["rank"],
            "momentum_pct": round(w["momentum_pct"], 2),
            "close": round(w["close"], 2),
            "held": sym in held,
            "updated": today,
            "tags": ["trading/stock"],
        }, f"""
# {sym}

**{w['status']}** · momentum rank #{w['rank']} of {len(watch)} · six-month return {pct(w['momentum_pct'])} · close {usd(w['close'])}
{('Left out of targets: ' + w['note'] + '. ') if w.get('note') else ''}Updated {stamp}

{f"Holding {pos['qty']:g} shares bought at {usd(pos['avg_cost'])}, now {usd(pos['price'])}: {signed(pos['unrealized_pl'])} ({pct(pos['unrealized_pl_pct'])})." if pos else 'Not held.'}

## Orders
{table(['Date', 'Side', 'Shares', 'Fill price', 'Status'],
       [[f"[[Daily/{o['date']}|{o['date']}]]", o['side'], f"{o['qty']:g}", usd(o['filled_price']), o['status']] for o in sym_orders])
 if sym_orders else 'No orders for this stock yet.'}

Back to [[Home base]] · [[Strategy]]
""", extra_if_new="\n## My notes\n\n")

    # Stocks you trade yourself outside the bot's list still get a note.
    for p in positions:
        if p["symbol"] not in {w["symbol"] for w in watch}:
            write_note(VAULT / "Stocks" / f"{p['symbol']}.md",
                       {"symbol": p["symbol"], "status": "Held (yours)", "held": True, "updated": today, "tags": ["trading/stock"]},
                       f"# {p['symbol']}\n\nYour own position, not managed by the bot: {p['qty']:g} shares at {usd(p['avg_cost'])}, "
                       f"now {usd(p['price'])}, {signed(p['unrealized_pl'])}.\n\nBack to [[Home base]]",
                       extra_if_new="\n## My notes\n\n")

    # --- Trade log ---
    write_note(VAULT / "Trade log.md", {"updated": stamp, "tags": ["trading"]}, f"""
# Trade log

Every order on the paper account, newest first. Updated {stamp}.

{table(['Date', 'Side', 'Stock', 'Shares', 'Fill price', 'Total', 'Status'],
       [[f"[[Daily/{o['date']}|{o['date']}]]", o['side'], stock_link(o['symbol']), f"{o['qty']:g}",
         usd(o['filled_price']), usd(o['value']) if o['value'] else '—', o['status']] for o in orders]) if orders else 'No orders yet.'}

## Completed trades
{table(['Stock', 'Bought', 'Sold', 'Profit / loss'],
       [[stock_link(t['symbol']), usd(t['buy_price']), usd(t['sell_price']), f"{signed(t['pl'])} ({pct(t['pl_pct'])})"] for t in closed])
 if closed else 'None yet. A trade completes when a stock the bot bought is sold.'}

Back to [[Home base]]
""")

    # --- Strategy ---
    write_note(VAULT / "Strategy.md", {"updated": stamp, "tags": ["trading"]}, f"""
# Strategy

The bot runs on [GitHub Actions]({REPO_URL}/actions) every 15 minutes and trades the paper account by fixed rules:

- **Stocks, monthly momentum:** rank the Dow 30 by their return over the six months to the end of last month. The top ten (see "Most stocks held at once") that are profitable with growing sales are the month's targets.
- **Buy** targets it doesn't hold, about the same dollar amount each (whole shares; see "Dollars per stock"), waiting out upcoming earnings; top up any target well below that size; **sell** stocks that drop out of the targets.
- **Crypto, daily trend (24/7):** hold a set dollar amount of Bitcoin and Ethereum while the 20-day average is clearly above the 50-day; sell when clearly below.
- **News sell (safety valve):** every run, sell a holding whose recent headlines score very badly. Bad-but-not-terrible news no longer blocks buys: a backtest showed those stocks went on to beat the average.
- **Rebound buy-back:** within 5 days of a news sale, buy back if the price climbs 3% above the sale price.
- Cash never drops below the floor, and positions you bought yourself are never touched.

Chosen after a backtest (Jan 2019 onward): monthly momentum returned +212% vs +186% for holding SPY and +73% for the old stock trend rules. Past results don't predict future ones.

## Current settings
{table(['Rule', 'Value'], [[s['setting'], s['value']] for s in settings])}

Code: [{REPO_URL}]({REPO_URL}) · Back to [[Home base]]
""")

    # --- Journal base (Obsidian Bases) ---
    base = VAULT / "Trading journal.base"
    if not base.exists():
        base.write_text("""filters:
  and:
    - file.inFolder("Daily")
views:
  - type: table
    name: Daily results
    order:
      - file.name
      - account_value
      - day_pl
      - total_pl
      - cash
      - stocks_held
      - orders_today
""")

    # --- Home base ---
    write_note(VAULT / "Home base.md", {"updated": stamp, "tags": ["trading"]}, f"""
# Trading bot · home base

> Paper account {acct['account']} · snapshot {stamp} · [Open dashboard]({DASHBOARD_URL})

| Account value | Today | Total since start | Cash | In stocks |
|---|---|---|---|---|
| {usd(acct['equity'])} | {signed(acct['day_pl'])} | {signed(acct['total_pl'])} ({pct(acct['total_pl_pct'])}) | {usd(acct['cash'])} | {usd(acct['invested'])} |

- **Realized profit:** {signed(realized)} · **Unrealized:** {signed(unrealized)} · **Win rate:** {win_rate}
- **Orders filled:** {len(filled)} · **Waiting to fill:** {len(waiting)}
- **Today's journal:** [[Daily/{today}|{today}]] · all days in [[Trading journal.base|Trading journal]]

## Holding now
{', '.join(stock_link(p['symbol']) for p in positions) or 'Nothing yet.'}

## This month's stock targets (top momentum)
{', '.join(f"{stock_link(w['symbol'])} {pct(w['momentum_pct'])}" for w in watch if w['status'] in ('Target', 'Held')) or 'None.'}
{('Left out: ' + ', '.join(f"{stock_link(w['symbol'])} ({w['note']})" for w in watch if w['status'] == 'Left out')) if any(w['status'] == 'Left out' for w in watch) else ''}

## Crypto
{', '.join(f"[[Crypto/{c['symbol'].replace('/', '-')}|{c['symbol']}]] {c['status'].lower()} ({pct(c['gap_pct'])})" for c in crypto) or 'Not trading crypto.'}

## Notes
[[Trade log]] · [[Strategy]] · [Bot code]({REPO_URL}) · [Bot runs]({REPO_URL}/actions)

## Second brain (trading wiki)
[[START-HERE|Start here]] · [[index|Wiki index]] · [[money-management]] · [[position-sizing-formulas-and-drawdown-math]] · [[golden-cross-death-cross]] · [[relative-strength-analysis]] · [[Trading Journal]]
""")

    print(f"Wrote notes to {VAULT}: Home base, Daily/{today}, {len(watch)} stock notes, Trade log, Strategy")


if __name__ == "__main__":
    main()
