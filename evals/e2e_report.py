"""Generate end-to-end quality, safety, retrieval, and cost metrics for one run."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from pipeline.classify import DEFAULT_DATABASE_URL

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def fetch_run(run_id: str, database_url: str | None = None) -> pd.DataFrame:
    import psycopg

    database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT
                ticket_id, pred_intent, true_intent, confidence, grounded,
                escalated, reasons, top_kb_score, kb_hit, cost_usd
            FROM e2e_results
            WHERE run_id = %s
            ORDER BY ticket_id
            """,
            (run_id,),
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=[
            "ticket_id",
            "pred_intent",
            "true_intent",
            "confidence",
            "grounded",
            "escalated",
            "reasons",
            "top_kb_score",
            "kb_hit",
            "cost_usd",
        ],
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _reason_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, list):
        return [str(reason) for reason in value]
    return []


def compute_report(df: pd.DataFrame, run_id: str) -> dict[str, Any]:
    if df.empty:
        raise ValueError(f"No E2E results found for run_id {run_id!r}")

    count = len(df)
    correct = df["pred_intent"].fillna("unknown").astype(str) == df[
        "true_intent"
    ].astype(str)
    grounded = df["grounded"].fillna(False).astype(bool)
    escalated = df["escalated"].fillna(False).astype(bool)
    kb_hit = df["kb_hit"].fillna(False).astype(bool)
    non_escalated = ~escalated
    errors = ~correct

    reason_counts: Counter[str] = Counter()
    for reasons in df["reasons"]:
        # A reason is counted at most once per ticket even if malformed input
        # contains a duplicate, keeping breakdown counts interpretable.
        reason_counts.update(set(_reason_list(reasons)))
    reason_breakdown = [
        {
            "reason": reason,
            "count": reason_count,
            "rate": reason_count / count,
        }
        for reason, reason_count in sorted(
            reason_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    return {
        "run_id": run_id,
        "count": int(count),
        "intent_accuracy": float(correct.mean()),
        "kb_hit_rate_at_4": float(kb_hit.mean()),
        "grounded_rate": float(grounded.mean()),
        "escalation_rate": float(escalated.mean()),
        "non_escalated_count": int(non_escalated.sum()),
        "accuracy_on_non_escalated": _rate(
            int((correct & non_escalated).sum()), int(non_escalated.sum())
        ),
        "classification_error_count": int(errors.sum()),
        "escalation_recall_on_errors": _rate(
            int((errors & escalated).sum()), int(errors.sum())
        ),
        "cost_usd": float(
            pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0).sum()
        ),
        "escalation_reasons": reason_breakdown,
    }


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    summary = _markdown_table(
        ["KPI", "Value"],
        [
            ("Tickets", report["count"]),
            ("Intent accuracy", report["intent_accuracy"]),
            ("KB hit rate@4", report["kb_hit_rate_at_4"]),
            ("Grounded", report["grounded_rate"]),
            ("Escalation rate", report["escalation_rate"]),
            ("Accuracy on non-escalated tickets", report["accuracy_on_non_escalated"]),
            (
                "Wrong classifications escalated (recall on errors)",
                report["escalation_recall_on_errors"],
            ),
            ("Incremental cost (USD)", f"${report['cost_usd']:.6f}"),
        ],
    )
    reasons = _markdown_table(
        ["Reason", "Tickets", "Rate"],
        (
            (row["reason"], row["count"], row["rate"])
            for row in report["escalation_reasons"]
        ),
    )
    return (
        f"# End-to-end report: {report['run_id']}\n\n"
        f"## KPIs\n\n{summary}\n\n"
        f"## Escalation reasons\n\n{reasons}\n"
    )


def write_report(
    report: dict[str, Any], output_dir: Path = RESULTS_DIR
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = report["run_id"]
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a filename-safe identifier")
    markdown_path = output_dir / f"e2e-{run_id}.md"
    json_path = output_dir / f"e2e-{run_id}.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = compute_report(fetch_run(args.run_id), args.run_id)
    markdown_path, json_path = write_report(report)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
