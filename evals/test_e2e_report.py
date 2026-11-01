"""Unit tests for E2E KPI aggregation without database or API calls."""

import pandas as pd

from evals.e2e_report import compute_report


def test_compute_report_kpis_and_reason_breakdown():
    frame = pd.DataFrame(
        [
            ("a", "a", True, False, [], True, 0.01),
            ("b", "c", False, True, ["low_confidence", "not_grounded"], False, 0.02),
            ("d", "x", True, False, [], True, 0.00),
            ("e", "e", False, True, ["not_grounded"], True, 0.00),
        ],
        columns=[
            "pred_intent",
            "true_intent",
            "grounded",
            "escalated",
            "reasons",
            "kb_hit",
            "cost_usd",
        ],
    )

    report = compute_report(frame, "unit-run")

    assert report["intent_accuracy"] == 0.5
    assert report["kb_hit_rate_at_4"] == 0.75
    assert report["grounded_rate"] == 0.5
    assert report["escalation_rate"] == 0.5
    assert report["accuracy_on_non_escalated"] == 0.5
    assert report["escalation_recall_on_errors"] == 0.5
    assert report["cost_usd"] == 0.03
    assert report["escalation_reasons"] == [
        {"reason": "not_grounded", "count": 2, "rate": 0.5},
        {"reason": "low_confidence", "count": 1, "rate": 0.25},
    ]
