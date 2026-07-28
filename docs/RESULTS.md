# v2 → v3 Changelog

This document records what changed between the v2 protocol (shipped in the
original STOG release) and the v3 protocol shipped in this repository, and why.

## 1. Split correction: chronological evaluation

- **v2**: EPF train/val/test blocks were drawn with a seeded shuffle
  (`seeded_shuffle`), which leaks future regime information into training and
  inflates router skill.
- **v3**: `src/data/epf.py` gains `split_mode="chronological"` (default).
  All v3 experiments run strictly forward in time; the shuffle mode remains
  only as an ablation.

## 2. Seed extension

- **v2**: single-seed headline numbers.
- **v3**: 5 seeds per market for the E6 routing benchmark. Headline tables
  report 5-seed mean ± std per market plus the pooled `ALL` row, with
  block-bootstrap TOST equivalence tests (`block=168, B=10000`), MASE against a
  naive-24h denominator, and Cohen's d effect sizes.

## 3. LEAR and MSTL enter the routing pool

- **v2**: the router chose among 19 deep experts only — a pool that excluded
  the strongest known EPF method class.
- **v3**: the pool is unified to **21 methods (19 deep experts + MSTL + LEAR)**.
  The router discovers LEAR on its own: it is the most-selected member at a
  **29.3%** test-window selection rate. Headline v3 numbers (ALL, test MSE):

  | Method | MSE |
  |---|---|
  | STOG-Router | 136.7 |
  | LEAR | 127.0 |
  | B1-deep | 158.4 |
  | Oracle | 94.8 |

## 4. Control arms redesigned (E10 placebo)

- **v2**: grafting gains were reported against static baselines only, leaving
  open the possibility that any per-window selection helps.
- **v3**: E10 adds a **placebo gate** (label-shuffled routing with identical
  selection frequency). The real gate beats the placebo by **−19.3% MSE at
  p = 1e-5**, isolating genuine probe→expert signal from selection noise.

## 5. Amplitude-equalized generator toggle

- Synthetic experiments (E1/E4 families) gain an amplitude-equalization switch
  on the field generator, separating "router responds to spectral/regime
  structure" from "router responds to trivial amplitude cues".

## 6. Standard Hedge control

- `analysis_v3_t4_hedge.py` / `analysis_v3_t4b_hedge_bound.py` add a standard
  exponentially-weighted Hedge forecaster as a routing control, plus a
  theoretical-bound comparison. Probe routing is benchmarked against this
  classical alternative rather than only against static experts.

## 7. Sentinel adjudication

- `analysis_v3_t1_sentinel.py` formalizes pre-registered falsification
  tripwires (sentinels) per experiment and adjudicates pass/fail against them.
  Known issue: one arm's bookkeeping miscounts (see README limitations); the
  adjudication outcomes themselves are unchanged.

## 8. E12 downgraded

- After protocol review, E12 (`run_e12_analysis.py`, `analysis_v3_t2_e12.py`)
  was downgraded from a confirmatory to an exploratory analysis; its outputs
  are reported as hypothesis-generating only.

## 9. Theorem 4 restricted version falsified (E4 v4)

- `run_e4_v4.py` tests the restricted ("limited-version") form of Theorem 4
  under the corrected protocol. The restricted version is **falsified**: a
  scalar regime-overlap gate does not predict routing benefit. The full dynamic
  window-conditioned gate survives; the paper's claims are scoped accordingly.

## 10. Failure analysis: sentinel / N10

- N10 stress condition in E8 carries a shape bug (invalid numbers, pending
  fix); sentinel accounting issues documented above. Both are listed as known
  limitations rather than silently patched, to keep the archived v3 outputs
  interpretable.
