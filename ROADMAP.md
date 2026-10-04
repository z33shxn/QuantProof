# Roadmap

Planned directions, in rough priority order. No dates are promised; items move when
contributors pick them up. Each item lists why it matters and what "done" would mean.

## Correctness and coverage

- **Cross-file static analysis.** Today data flow (L3) is intra-file. Follow imports of
  local modules so helpers in `utils.py` are summarised like same-file helpers. Done when
  the adversarial suite includes multi-file cases.
- **Container mutation in the taint analysis.** Track `list.append`, `dict.update` and
  similar mutations (currently a documented boundary).
- **Point-in-time data checks.** Optional availability timestamps per column (e.g.
  fundamentals released after period end) so validation can flag values used before they
  were published.
- **HAC standard errors for SPA.** Offer a kernel estimator of `ω_k` alongside the
  bootstrap estimate and report both.
- **Per-symbol results for panels.** Per-symbol performance, cost attribution and
  causality evidence tables.
- **Calendars for panels.** Optional alignment of symbols with different trading
  calendars instead of only flagging them.

## Usability

- **Engine adapters as separate packages** (VectorBT, Backtrader, LEAN) maintained with
  tests against pinned engine versions, outside the core.
- **Pre-commit hook / GitHub Action** wrapping `quantproof scan` and `quantproof audit`
  with Markdown output for PR comments.
- **Sandboxed execution** of strategy code (subprocess with resource limits) as an opt-in.
- **PyPI release** once the API has been exercised by early users.

## Research

- **Combinatorial selection beyond grids**: support declared random/Bayesian searches
  with their full trial histories for DSR/PBO.
- **Effective number of trials**: compare Li & Ji with clustering-based estimates and
  document when each is appropriate.
