# Claude headline scoring test (2026-11-19)

300 random Alpaca headlines on the Dow 30 (Jul 1 – Nov 19, 2026, after the model's training cutoff;
price-move recaps removed). Each was labelled good / bad / unknown for the tagged company's stock in
the short term (Lopez-Lira & Tang's question). Returns are market-adjusted (minus SPY), from the open
of the first session after publication.

| Score | Horizon | Good | Bad | Spread | p |
|---|---|---|---|---|---|
| Claude | 1 day | −0.32% (26) | +0.32% (37) | −0.64 pts | ≈0.11 |
| Claude | 5 days | −1.62% (25) | +0.05% (36) | −1.66 pts | ≈0.12 |
| Keyword | 5 days | −1.99% (9) | +2.19% (23) | −4.17 pts | ≈0.02 |

Verdict: no positive news edge in Dow mega-caps; if anything, short-term reversal. `scored.csv` holds
every headline, label and return; `labels.json` the label ids.
