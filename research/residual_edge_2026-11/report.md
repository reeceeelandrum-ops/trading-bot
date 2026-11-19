# Residual Edge Analysis

**Status:** BASELINE_EXPLAINED
**Decision eligibility:** REVIEW_REQUIRED
**Period:** 2017-01-17 to 2026-11-19 (2476 observations)
**Primary model:** market

## Primary Model

| Metric | Value |
|---|---:|
| R² | 0.7517 |
| Adjusted R² | 0.7516 |
| Annualized alpha | 2.8032% |
| Alpha HAC t-stat | 0.954 |
| Annualized residual volatility | 9.3917% |
| Residual edge ratio | 0.298 |

## Baseline Loadings

| Baseline | Loading | HAC t-stat | VIF |
|---|---:|---:|---:|
| market_return | 0.8995 | 31.273 | 1.000 |

## Model Sensitivity

| Model | Role | R² | Annualized alpha | Residual edge ratio | Status |
|---|---|---:|---:|---:|---|
| market | primary | 0.7517 | 2.8032% | 0.298 | BASELINE_EXPLAINED |
| equal_weight_dow | sensitivity | 0.7931 | 0.6658% | 0.078 | BASELINE_EXPLAINED |

## Evidence Warnings

- **HIGH — SURVIVORSHIP_BIAS_RISK:** A current-constituent baseline is not point-in-time and can introduce survivorship bias.
- **HIGH — RETURN_BASIS_MISMATCH:** Strategy and baseline returns use different gross/net cost bases.
- **HIGH — NOT_OUT_OF_SAMPLE:** Treat in-sample residual edge as exploratory until confirmed out of sample.

## Regime Breakdown

### volatility_regime

| Regime | N | Evidence | Annualized active return | Active Sharpe |
|---|---:|---|---:|---:|
| high | 1228 | ADEQUATE | 3.5624% | 0.329 |
| low | 1248 | ADEQUATE | 2.0561% | 0.266 |

## Interpretation

- Declared baselines explain most variation and residual edge is weak.
- This report does not authorize a trade or an automatic exposure change.
- Confirm exploratory findings on untouched out-of-sample or live data.
