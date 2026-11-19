"""Rule-based checks on upcoming earnings, company figures, and recent news.

Every check is a fixed, transparent rule with thresholds from config.json.
Missing data never blocks a trade on its own; it's reported as "unknown".
"""

import os
import re
from datetime import date, datetime, timedelta, timezone

import yfinance as yf

# ---------- upcoming earnings ----------

def next_earnings_date(symbol: str) -> date | None:
    try:
        dates = yf.Ticker(symbol).calendar.get("Earnings Date") or []
    except Exception:
        return None
    upcoming = [d for d in dates if d >= date.today()]
    return min(upcoming) if upcoming else None


def earnings_check(symbol: str, blackout_days: int) -> tuple[bool, str]:
    d = next_earnings_date(symbol)
    if d is None:
        return True, "earnings date unknown"
    days = (d - date.today()).days
    if days <= blackout_days:
        return False, f"earnings in {days} day(s) on {d}"
    return True, f"earnings {d}"


# ---------- company figures ----------

def company_check(symbol: str, require_profitable: bool, min_revenue_growth_pct: float) -> tuple[bool, str]:
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return True, "company data unknown"
    margin, growth = info.get("profitMargins"), info.get("revenueGrowth")
    notes = []
    if margin is None:
        notes.append("margin unknown")
    else:
        notes.append(f"margin {margin*100:.1f}%")
        if require_profitable and margin <= 0:
            return False, f"not profitable (margin {margin*100:.1f}%)"
    if growth is None:
        notes.append("revenue growth unknown")
    else:
        notes.append(f"revenue growth {growth*100:+.1f}%")
        if growth * 100 < min_revenue_growth_pct:
            return False, f"revenue growth {growth*100:+.1f}% below {min_revenue_growth_pct:+g}%"
    return True, ", ".join(notes)


# ---------- recent news ----------

NEGATIVE = [
    "downgrade", "downgraded", "downgrades", "lawsuit", "sues", "sued", "investigation", "probe",
    "subpoena", "fraud", "recall", "recalls", "bankruptcy", "layoffs", "lays off", "job cuts",
    "misses", "missed estimates", "cuts guidance", "lowers guidance", "lowered guidance",
    "cuts outlook", "lowers outlook", "warns", "profit warning", "plunge", "plunges", "tumbles",
    "slumps", "sinks", "halted", "resigns", "data breach", "antitrust", "fined", "penalty",
    "delisting", "short seller", "accounting irregularities", "restatement",
]
POSITIVE = [
    "upgrade", "upgraded", "upgrades", "beats", "beat estimates", "tops estimates",
    "raises guidance", "raised guidance", "raises outlook", "raises forecast", "record revenue",
    "record profit", "fda approval", "approved", "buyback", "raises dividend", "dividend increase",
    "surges", "soars", "jumps", "rallies", "wins contract", "awarded contract",
]
_NEG = [re.compile(rf"\b{re.escape(w)}\b") for w in NEGATIVE]
_POS = [re.compile(rf"\b{re.escape(w)}\b") for w in POSITIVE]
_TARGET = re.compile(r"price target.*?to \$?([\d,.]+)\s+from \$?([\d,.]+)", re.I)


def headline_score(headline: str) -> int:
    """-1, 0, or +1 per headline: negative words, positive words, and price-target moves."""
    h = headline.lower()
    score = sum(bool(p.search(h)) for p in _POS) - sum(bool(p.search(h)) for p in _NEG)
    m = _TARGET.search(headline)
    if m:
        new, old = (float(x.replace(",", "").rstrip(".")) for x in m.groups())
        score += (new > old) - (new < old)
    return max(-1, min(1, score))


class NewsScorer:
    def __init__(self, lookback_days: int):
        from alpaca.data.historical.news import NewsClient

        key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        self.client = NewsClient(key, secret) if key and secret and not key.startswith("your_") else None
        self.lookback_days = lookback_days
        self.cache: dict[str, tuple[int, list[str]]] = {}

    def score(self, symbol: str) -> tuple[int | None, list[str]]:
        """Sum of headline scores over the lookback window, plus the scored headlines."""
        if "/" in symbol:  # crypto: Alpaca's feed has none, so use Yahoo Finance headlines
            return self._crypto_score(symbol)
        if self.client is None:
            return None, []
        if symbol not in self.cache:
            from alpaca.data.requests import NewsRequest
            try:
                start = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
                items = self.client.get_news(NewsRequest(symbols=symbol, start=start, limit=50)).data["news"]
            except Exception:
                return None, []
            # Only count stories mainly about this company, not roundups naming many tickers.
            items = [n for n in items if len(n.symbols) <= 3]
            scored = [(headline_score(n.headline), n.headline) for n in items]
            total = sum(s for s, _ in scored)
            self.cache[symbol] = (total, [f"{s:+d} {h}" for s, h in scored if s])
        return self.cache[symbol]

    def _crypto_score(self, symbol: str) -> tuple[int | None, list[str]]:
        if symbol not in self.cache:
            try:
                items = yf.Ticker(symbol.replace("/", "-")).news or []
            except Exception:
                return None, []
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
            scored = []
            for item in items:
                c = item.get("content", item)
                title, when = c.get("title") or "", c.get("pubDate")
                try:
                    if when and datetime.fromisoformat(when.replace("Z", "+00:00")) < cutoff:
                        continue
                except ValueError:
                    pass
                scored.append((headline_score(title), title))
            total = sum(s for s, _ in scored)
            self.cache[symbol] = (total, [f"{s:+d} {h}" for s, h in scored if s])
        return self.cache[symbol]
