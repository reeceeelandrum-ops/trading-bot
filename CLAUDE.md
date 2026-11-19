# CLAUDE.md — trading bot rulebook

Read this before changing anything. This repo runs a rule-based bot on an **Alpaca paper account**
every 15 minutes via GitHub Actions. It is public: never commit keys, account numbers, balances or
personal paths.

## What the bot does (source of truth: `config.json` + `bot.py`)
- **Stocks:** monthly momentum. Rank the `universe` by return over `momentum_lookback_days` to the end
  of last month; the top `max_positions` that pass the company check are the month's targets.
  Equal dollars per stock (`stock_dollars: "auto"`), top up below 80%, trim above 125%,
  sell stocks that leave the targets. Buys wait out earnings (`earnings_blackout_days`).
  The news buy-block (`news_block_buy_score`) is **off** (null) since 2026-11-19: it skipped stocks
  that then beat the average (`news_backtest.py`).
- **Crypto:** `crypto.symbols` on a daily SMA `short_window`/`long_window` trend with `buffer_pct`.
- **Every run:** very-bad-news sell (`news_sell_score`, a rare safety valve with no proven edge), rebound buy-back (`reentry_*`), cash never below
  `min_cash`, never touch positions outside the bot's lists, at most one buy and one sell per
  symbol per day.

## How changes are made (the quant loop)
1. **State the hypothesis** and where it comes from (paper, book, observation).
2. **Backtest it** against the live rules and buy-and-hold SPY: `backtest.py`, `backtest_grid.py`,
   `backtest_volscale.py`, `news_backtest.py`. Signals use completed bars only; fills at next open;
   include slippage. Report return, worst drop, return/risk, trades, and both test periods.
3. **Count your trials.** Every variant tried is a draw from noise; prefer settings that work across
   neighbouring parameters and periods over the single best cell (Bailey & López de Prado, deflated
   Sharpe / probability of backtest overfitting). Record each experiment in the experiment log.
4. **Adopt only what holds up**, then paper-trade it and review; never tune on the last few weeks.
5. **Verify against the broker.** After any live change, confirm orders and positions via the Alpaca
   API, not just the bot's log.

## Already tested — don't re-propose without new evidence
SPY golden-cross and 200-day regime filters; 15–25% trailing stops; 12-month and 12-1 momentum
(no better than 6-month); 5 vs 8 vs 10 vs 15 positions (10 chosen); crypto 50/200 cross;
volatility-managed momentum (12–20% targets); keyword news buy-block (hurt). Details in the experiment log.

## Lessons from other AI trading agents (2026-11 research)
- Live LLM trading contests (Alpha Arena season 1, LiveTradeBench) mostly lost money; the losers
  over-traded and over-leveraged, and benchmark-smart models were not better traders. Keep the LLM
  as researcher/engineer, keep decisions in tested rules.
- Multi-agent "trading firm" designs (TradingAgents) report strong short-window results that others
  question (look-ahead from pretraining). Useful idea: an independent risk check with veto power.

## Safety
- `TradingClient(..., paper=True)` only. Never switch to a live account.
- Third-party skills or scripts that touch keys must be read in full before use.
- Commit with the GitHub no-reply identity configured in this repo.
