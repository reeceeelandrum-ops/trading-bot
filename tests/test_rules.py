"""Rule tests. They run before every trading run on GitHub: if one fails, the bot doesn't trade.

    .venv/bin/python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import unittest
from pathlib import Path

import pandas as pd

from bot import blocks_buy, momentum_ranking, plan_crypto, plan_stocks, stock_dollars
from signals import headline_score

ROOT = Path(__file__).resolve().parent.parent
OK = lambda s: (True, "ok")  # noqa: E731
NEWS0 = lambda s: 0  # noqa: E731


def cfg(**kw):
    base = dict(stock_dollars=7000, max_positions=10, min_cash=20000, buffer_pct=1.0,
                crypto={"dollars": 1000, "symbols": ["BTC/USD", "ETH/USD"]},
                news_sell_score=-4, news_block_buy_score=None, reentry_rebound_pct=3, reentry_days=5)
    base.update(kw)
    return base


def plan(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):  # keep test output clean
        return plan_stocks(*a, **k)


def summary(trades):
    return [(t[0], t[1], t[2]) for t in trades]


class Config(unittest.TestCase):
    def test_live_config_is_sane(self):
        c = json.loads((ROOT / "config.json").read_text())
        self.assertTrue(c["universe"] and c["max_positions"] >= 1)
        self.assertGreaterEqual(c["min_cash"], 0)
        self.assertLess(c["short_window"], c["long_window"])
        self.assertIn(c["stock_dollars"], ("auto",)) if isinstance(c["stock_dollars"], str) else None
        self.assertLess(c["news_sell_score"], 0)

    def test_hands_off_positions_are_never_traded(self):
        from bot import load_config
        c = load_config()
        for sym in c["hands_off"]:
            self.assertNotIn(sym, c["universe"])  # never a target, never sold or trimmed
        self.assertIn("NVDA", c["hands_off"])
        # A hands-off stock held in the account is ignored even if the plan sees it held.
        t = plan(["A"], {"A"}, {"A": 70, "NVDA": 99}, {"A": 100.0, "NVDA": 151.0}, 1e5, cfg(), OK, NEWS0, {})
        self.assertNotIn("NVDA", [x[1] for x in t])

    def test_universe_size_is_deliberate(self):
        # The investable list must only change on purpose. If hands_off or the universe changes,
        # update EXPECTED here after re-running the backtests (2026-11-20 lesson).
        from bot import load_config
        c = load_config()
        raw = json.loads((ROOT / "config.json").read_text())
        EXPECTED = 29  # Dow 30 minus NVDA (on hands_off since 2026-11-19)
        self.assertEqual(len(c["universe"]), EXPECTED,
                         f"universe is {len(c['universe'])} stocks; hands_off={raw.get('hands_off')} — re-backtest and update EXPECTED")

    def test_paper_only(self):
        self.assertIn("paper=True", (ROOT / "bot.py").read_text())


class Sizing(unittest.TestCase):
    def test_auto_dollars(self):
        c = cfg(stock_dollars="auto")
        self.assertAlmostEqual(stock_dollars(c, 100_000), (100_000 - 20_000 - 2_000) / 10)
        self.assertEqual(stock_dollars(c, 10_000), 0.0)  # never negative

    def test_buys_whole_shares_by_dollars(self):
        t = plan(["A", "B"], {"A", "B"}, {}, {"A": 100.0, "B": 450.0}, 1e5, cfg(), OK, NEWS0, {})
        self.assertEqual(summary(t), [("buy", "A", 70), ("buy", "B", 15)])

    def test_share_price_above_budget_is_skipped(self):
        self.assertEqual(plan(["A"], {"A"}, {}, {"A": 9000.0}, 1e5, cfg(), OK, NEWS0, {}), [])

    def test_cash_floor_never_breached(self):
        t = plan(["A", "B"], {"A", "B"}, {}, {"A": 100.0, "B": 100.0}, 25_000, cfg(), OK, NEWS0, {})
        spent = sum(q * 100 for s, _, q, _ in [(x[0], x[1], x[2], x[3]) for x in t] if s == "buy")
        self.assertLessEqual(spent, 5_000)

    def test_top_up_and_trim(self):
        self.assertEqual(summary(plan(["A"], {"A"}, {"A": 10}, {"A": 100.0}, 1e5, cfg(), OK, NEWS0, {})), [("buy", "A", 60)])
        self.assertEqual(summary(plan(["A"], {"A"}, {"A": 60}, {"A": 100.0}, 1e5, cfg(), OK, NEWS0, {})), [])
        self.assertEqual(summary(plan(["A"], {"A"}, {"A": 100}, {"A": 100.0}, 1e5, cfg(), OK, NEWS0, {})), [("sell", "A", 30)])

    def test_trim_proceeds_fund_buys(self):
        t = plan(["A", "B"], {"A", "B"}, {"A": 140}, {"A": 100.0, "B": 100.0}, 20_000, cfg(), OK, NEWS0, {})
        self.assertIn(("buy", "B", 70), summary(t))

    def test_dry_run_unlimited_cash_does_not_crash(self):
        plan(["A"], {"A"}, {}, {"A": 100.0}, float("inf"), cfg(), OK, NEWS0, {})


class Rules(unittest.TestCase):
    def test_sell_non_targets_but_never_outside_positions(self):
        t = summary(plan(["B"], {"A", "B"}, {"A": 70, "MU": 1}, {"A": 100.0, "B": 100.0}, 1e5, cfg(), OK, NEWS0, {}))
        self.assertIn(("sell", "A", 70), t)
        self.assertNotIn("MU", [x[1] for x in t])

    def test_earnings_check_delays_buy(self):
        t = plan(["A", "B"], {"A", "B"}, {}, {"A": 100.0, "B": 100.0}, 1e5, cfg(),
                 lambda s: (s != "A", "earnings soon"), NEWS0, {})
        self.assertEqual([x[1] for x in t], ["B"])

    def test_news_sell_and_buy_block_switch(self):
        self.assertEqual(summary(plan(["A"], {"A"}, {"A": 70}, {"A": 100.0}, 1e5, cfg(), OK, lambda s: -5, {})), [("sell", "A", 70)])
        self.assertEqual(len(plan(["A"], {"A"}, {}, {"A": 100.0}, 1e5, cfg(), OK, lambda s: -3, {})), 1)  # block off
        self.assertEqual(plan(["A"], {"A"}, {}, {"A": 100.0}, 1e5, cfg(news_block_buy_score=-2), OK, lambda s: -3, {}), [])
        self.assertFalse(blocks_buy(None, cfg(news_block_buy_score=-2)))

    def test_rebound_buyback(self):
        self.assertEqual(plan(["A"], {"A"}, {}, {"A": 102.0}, 1e5, cfg(), OK, lambda s: -3, {"A": 100.0}), [])
        self.assertEqual(len(plan(["A"], {"A"}, {}, {"A": 103.5}, 1e5, cfg(), OK, lambda s: -3, {"A": 100.0})), 1)

    def test_crypto(self):
        ct = pd.DataFrame({"gap_pct": [5.0, -3.0]}, index=["BTC/USD", "ETH/USD"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual([(t[0], t[1]) for t in plan_crypto(ct, {}, {}, 1e5, cfg(), NEWS0, {})], [("buy", "BTC/USD")])
            plan_crypto(ct, {}, {}, 20_500, cfg(), NEWS0, {})
        self.assertNotIn("ETH/USD", out.getvalue())  # a downtrend coin never reports a cash problem
        self.assertEqual([(t[0], t[1]) for t in plan_crypto(ct, {"ETHUSD": 0.5}, {}, 1e5, cfg(), NEWS0, {})][-1], ("sell", "ETH/USD"))

    def test_momentum_uses_last_month_end_only(self):
        idx = pd.bdate_range("2026-01-01", "2026-03-20")
        df = pd.DataFrame({"A": range(1, len(idx) + 1), "B": [10.0] * len(idx)}, index=idx, dtype=float)
        df.loc["2026-03-02":, "B"] = 1000.0  # a March spike must not affect March's ranking
        r = momentum_ranking(df, 20, pd.Timestamp("2026-03-15").date())
        self.assertEqual(r.index[0], "A")


class Scanner(unittest.TestCase):
    def test_grade_enters_next_open_and_signs_by_direction(self):
        from scanner_eval import grade
        days = pd.bdate_range("2026-11-16", periods=8)
        a_open = pd.Series([100, 100, 100, 110, 111, 112, 113, 114.0], index=days)
        a_close = pd.Series([100, 100, 110, 111, 112, 113, 114, 121.0], index=days)
        s_open = pd.Series([500.0] * 8, index=days)
        s_close = pd.Series([500, 500, 500, 505, 505, 505, 505, 510.0], index=days)
        found = pd.Timestamp("2026-11-18 10:00")  # during the session -> enters next day's open (110)
        a, s, x = grade(found, 5, "up", a_open, a_close, s_open, s_close)
        self.assertAlmostEqual(a, 121 / 110 - 1)   # never credits the move that already happened
        self.assertAlmostEqual(s, 510 / 500 - 1)
        self.assertAlmostEqual(x, a - s)
        _, _, x_down = grade(found, 5, "down", a_open, a_close, s_open, s_close)
        self.assertAlmostEqual(x_down, -(a - s))
        self.assertIsNone(grade(found, 6, "up", a_open, a_close, s_open, s_close))  # horizon not reached


class News(unittest.TestCase):
    def test_headline_score(self):
        self.assertEqual(headline_score("UBS Adjusts Price Target on Home Depot to $417 From $420"), -1)
        self.assertEqual(headline_score("Goldman upgrades Visa to Buy"), 1)
        self.assertEqual(headline_score("Company defines new strategy"), 0)  # 'fine' inside 'defines' is not a match


if __name__ == "__main__":
    unittest.main()
