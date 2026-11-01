"""Export E2E evaluation results and knowledge base into a static JSON payload for the demo site."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

DEFAULT_DB_URL = "postgresql://support:support@localhost:5433/support"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "demo.json"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"


def load_kpis_from_disk(run_id: str) -> dict[str, Any] | None:
    """Attempt to locate precomputed evaluation KPIs in evals/results."""
    candidates = [
        RESULTS_DIR / f"e2e-{run_id}.json",
        RESULTS_DIR / f"{run_id}.json",
    ]
    if run_id.startswith("e2e-"):
        short_id = run_id[4:]
        candidates.append(RESULTS_DIR / f"e2e-{short_id}.json")
        candidates.append(RESULTS_DIR / f"{short_id}.json")

    for candidate in candidates:
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def compute_kpis_from_rows(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute standard KPIs if evals JSON file does not exist."""
    total = len(rows)
    if total == 0:
        return {"run_id": run_id, "count": 0}

    correct_count = sum(1 for r in rows if r["correct"])
    kb_hit_count = sum(1 for r in rows if r["kb_hit"])
    grounded_count = sum(1 for r in rows if r["grounded"])
    escalated_count = sum(1 for r in rows if r["escalated"])
    non_escalated = [r for r in rows if not r["escalated"]]
    non_escalated_count = len(non_escalated)
    correct_non_escalated = sum(1 for r in non_escalated if r["correct"])

    errors = [r for r in rows if not r["correct"]]
    classification_error_count = len(errors)
    escalated_errors = sum(1 for r in errors if r["escalated"])

    reasons_counter = Counter()
    for r in rows:
        reasons = r.get("reasons") or []
        for reason in reasons:
            reasons_counter[reason] += 1

    reason_breakdown = [
        {"reason": reason, "count": count, "rate": round(count / total, 4)}
        for reason, count in reasons_counter.most_common()
    ]

    return {
        "run_id": run_id,
        "count": total,
        "intent_accuracy": round(correct_count / total, 4),
        "kb_hit_rate_at_4": round(kb_hit_count / total, 4),
        "grounded_rate": round(grounded_count / total, 4),
        "escalation_rate": round(escalated_count / total, 4),
        "non_escalated_count": non_escalated_count,
        "accuracy_on_non_escalated": round(correct_non_escalated / non_escalated_count, 4)
        if non_escalated_count
        else 0.0,
        "classification_error_count": classification_error_count,
        "escalation_recall_on_errors": round(escalated_errors / classification_error_count, 4)
        if classification_error_count
        else 0.0,
        "escalation_reasons": reason_breakdown,
    }


def fetch_kb_articles(conn: psycopg.Connection) -> dict[str, dict[str, str]]:
    """Fetch knowledge base articles formatted as KB-xx: {title, intent}."""
    kb_dict: dict[str, dict[str, str]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, title, intent FROM kb_articles ORDER BY id")
        for row in cur.fetchall():
            kb_id = f"KB-{row['id']:02d}"
            kb_dict[kb_id] = {
                "title": row["title"],
                "intent": row["intent"],
            }
    return kb_dict


def fetch_run_tickets(conn: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    """Fetch ticket execution details for a given run_id joined with tickets."""
    query = """
        SELECT
            e.ticket_id,
            t.instruction AS text,
            t.category,
            e.true_intent,
            e.pred_intent,
            (e.pred_intent = e.true_intent) AS correct,
            e.confidence,
            e.reply,
            e.citations,
            e.grounded,
            e.escalated,
            e.reasons,
            e.top_kb_score,
            e.kb_hit
        FROM e2e_results e
        JOIN tickets t ON t.id = e.ticket_id
        WHERE e.run_id = %s
        ORDER BY e.ticket_id
    """
    tickets: list[dict[str, Any]] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (run_id,))
        for row in cur.fetchall():
            citations = row["citations"]
            if citations is None:
                citations = []
            elif not isinstance(citations, list):
                try:
                    citations = json.loads(citations)
                except Exception:
                    citations = []

            reasons = row["reasons"]
            if reasons is None:
                reasons = []
            elif not isinstance(reasons, list):
                try:
                    reasons = json.loads(reasons)
                except Exception:
                    reasons = []

            tickets.append({
                "ticket_id": row["ticket_id"],
                "text": row["text"],
                "category": row["category"],
                "true_intent": row["true_intent"],
                "pred_intent": row["pred_intent"],
                "correct": bool(row["correct"]),
                "confidence": float(row["confidence"]) if row["confidence"] is not None else 0.0,
                "reply": row["reply"] or "",
                "citations": citations,
                "grounded": bool(row["grounded"]),
                "escalated": bool(row["escalated"]),
                "reasons": reasons,
                "top_kb_score": float(row["top_kb_score"]) if row["top_kb_score"] is not None else None,
                "kb_hit": bool(row["kb_hit"]),
            })
    return tickets


def export_demo(
    runs: list[str],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    database_url: str | None = None,
) -> None:
    db_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(db_url) as conn:
        kb_data = fetch_kb_articles(conn)

        runs_data: dict[str, dict[str, Any]] = {}
        for run_id in runs:
            tickets = fetch_run_tickets(conn, run_id)
            kpis = load_kpis_from_disk(run_id)
            if not kpis:
                kpis = compute_kpis_from_rows(run_id, tickets)

            runs_data[run_id] = {
                "kpis": kpis,
                "tickets": tickets,
            }

    payload = {
        "runs": runs_data,
        "kb": kb_data,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Exported {len(runs)} run(s) and {len(kb_data)} KB article(s) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run IDs to export (e.g. e2e-v1 e2e-v2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Target output JSON path (default: docs/data/demo.json)",
    )
    args = parser.parse_args()
    export_demo(args.runs, args.output)


if __name__ == "__main__":
    main()
