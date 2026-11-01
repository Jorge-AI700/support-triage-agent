"""Unit tests for E2E KPI aggregation without database or API calls."""

import pandas as pd
import pytest

from evals.e2e_report import (
    compute_comparison,
    compute_report,
    render_comparison_markdown,
    write_comparison,
)


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


def test_compare_reports_renders_kpi_deltas_and_expected_filename(tmp_path):
    before = {
        "run_id": "e2e-v1",
        "count": 200,
        "intent_accuracy": 0.91,
        "kb_hit_rate_at_4": 0.80,
        "grounded_rate": 0.83,
        "escalation_rate": 0.20,
        "accuracy_on_non_escalated": 0.90,
        "escalation_recall_on_errors": 0.40,
        "cost_usd": 0.20,
    }
    after = {
        **before,
        "run_id": "e2e-v2",
        "intent_accuracy": 0.95,
        "grounded_rate": 0.96,
        "escalation_recall_on_errors": 0.75,
        "cost_usd": 0.24,
    }

    comparison = compute_comparison(before, after)
    markdown = render_comparison_markdown(comparison)
    path = write_comparison(comparison, tmp_path)

    intent_row = next(
        row for row in comparison["rows"] if row["key"] == "intent_accuracy"
    )
    assert intent_row["delta"] == pytest.approx(0.04)
    assert "| Intent accuracy | 0.9100 | 0.9500 | +0.0400 |" in markdown
    assert path.name == "compare-e2e-v1-vs-e2e-v2.md"
    assert path.read_text(encoding="utf-8") == markdown
