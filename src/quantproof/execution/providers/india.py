"""Reference cost provider for Indian equity markets (NSE), effective-dated.

**This is an illustrative reference, not tax or legal advice.** Statutory rates
change by Finance Act, exchange circular and state law; brokerage depends on the
broker. Verify every rate against primary sources (Finance Act / CBDT notifications
for STT, NSE circulars for transaction charges, SEBI for turnover fees, the Indian
Stamp Act as amended in 2020) before relying on it.

Coverage starts 2024-10-01 (the date NSE moved to uniform, non-slab transaction
charges under SEBI's "true to label" circular and the Finance (No. 2) Act 2024 STT
changes on F&O took effect). Earlier dates raise an error.

Rates (fractions of traded value; options use premium value):

=================  ===========================  ==========  ========  ==========
segment            STT                          NSE txn     SEBI      stamp (buy)
=================  ===========================  ==========  ========  ==========
equity_delivery    0.1% buy and sell            0.00297%    0.0001%   0.015%
equity_intraday    0.025% sell                  0.00297%    0.0001%   0.003%
equity_futures     0.02% sell; 0.05% from       0.00173%    0.0001%   0.002%
                   2026-04-01 (Budget 2026-27)
equity_options     0.1% of premium sell; 0.15%  0.03503%    0.0001%   0.003%
                   from 2026-04-01
=================  ===========================  ==========  ========  ==========

GST of 18% applies to brokerage + exchange transaction charges + SEBI fees.
Option exercise STT (charged on intrinsic value) is not modelled.
"""

from __future__ import annotations

import datetime as dt

from quantproof.execution.providers.base import CostProvider, FeeSchedule

_START = dt.date(2024, 10, 1)
_BUDGET_2026 = dt.date(2026, 4, 1)
_SEBI = 10 / 1e7  # ₹10 per crore
_GST = 0.18

SOURCES = [
    "SEBI circular on uniform ('true to label') exchange charges, effective 2024-10-01; NSE "
    "transaction charges as reported in https://www.business-standard.com/markets/news/"
    "higher-stt-and-transaction-charges-on-nse-bse-mcx-effective-today-124100100146_1.html",
    "Finance (No. 2) Act 2024: STT on futures 0.02%, options premium 0.1% from 2024-10-01",
    "Union Budget 2026-27: STT on futures 0.05%, options premium 0.15% from 2026-04-01, as "
    "reported in https://www.5paisa.com/news/stt-hike-on-fo-to-take-effect-from-april-1-amid-"
    "rising-options-activity and https://www.icicidirect.com/ilearn/futures-and-options/"
    "articles/stt-changes-in-budget-2026-what-f-o-traders-should-know",
    "Indian Stamp Act (as amended, effective 2020-07-01): uniform stamp duty on the buy side",
]


def _schedules() -> list[FeeSchedule]:
    s: list[FeeSchedule] = [
        FeeSchedule(
            "equity_delivery",
            _START,
            None,
            buy_tax=0.001,
            sell_tax=0.001,
            exchange_fee=0.0000297,
            regulatory_fee=_SEBI,
            stamp_duty_buy=0.00015,
            tax_on_fees=_GST,
        ),
        FeeSchedule(
            "equity_intraday",
            _START,
            None,
            sell_tax=0.00025,
            exchange_fee=0.0000297,
            regulatory_fee=_SEBI,
            stamp_duty_buy=0.00003,
            tax_on_fees=_GST,
        ),
        FeeSchedule(
            "equity_futures",
            _START,
            _BUDGET_2026 - dt.timedelta(days=1),
            sell_tax=0.0002,
            exchange_fee=0.0000173,
            regulatory_fee=_SEBI,
            stamp_duty_buy=0.00002,
            tax_on_fees=_GST,
        ),
        FeeSchedule(
            "equity_futures",
            _BUDGET_2026,
            None,
            sell_tax=0.0005,
            exchange_fee=0.0000173,
            regulatory_fee=_SEBI,
            stamp_duty_buy=0.00002,
            tax_on_fees=_GST,
            notes="STT raised by Union Budget 2026-27",
        ),
        FeeSchedule(
            "equity_options",
            _START,
            _BUDGET_2026 - dt.timedelta(days=1),
            sell_tax=0.001,
            exchange_fee=0.0003503,
            regulatory_fee=_SEBI,
            stamp_duty_buy=0.00003,
            tax_on_fees=_GST,
            notes="Rates apply to premium value",
        ),
        FeeSchedule(
            "equity_options",
            _BUDGET_2026,
            None,
            sell_tax=0.0015,
            exchange_fee=0.0003503,
            regulatory_fee=_SEBI,
            stamp_duty_buy=0.00003,
            tax_on_fees=_GST,
            notes="Rates apply to premium value; STT raised by Union Budget 2026-27",
        ),
    ]
    return s


def india_nse_provider() -> CostProvider:
    """Effective-dated NSE equity/F&O statutory charges (reference; verify before use)."""
    return CostProvider(
        name="india_nse",
        jurisdiction="IN",
        schedules=_schedules(),
        sources=SOURCES,
        last_reviewed="2026-10-04",
        disclaimer=(
            "Reference values for research only. Rates depend on instrument, exchange, broker, "
            "transaction type, jurisdiction and date; verify against primary sources."
        ),
    )
