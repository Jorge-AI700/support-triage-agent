"""Generate classification metrics for one prediction run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from pipeline.classify import ALLOWED_INTENTS, DEFAULT_DATABASE_URL

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def fetch_run(run_id: str, database_url: str | None = None) -> pd.DataFrame:
    import psycopg

    database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT
                p.ticket_id,
                p.model,
                t.intent AS true_intent,
                p.pred_intent,
                t.category AS true_category,
                p.pred_category,
                p.confidence
            FROM predictions AS p
            JOIN tickets AS t ON t.id = p.ticket_id
            WHERE p.run_id = %s
            ORDER BY p.ticket_id, p.id
            """,
            (run_id,),
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=[
            "ticket_id",
            "model",
            "true_intent",
            "pred_intent",
            "true_category",
            "pred_category",
            "confidence",
        ],
    )


def compute_report(df: pd.DataFrame, run_id: str) -> dict[str, Any]:
    if df.empty:
        raise ValueError(f"No predictions found for run_id {run_id!r}")

    true_intent = df["true_intent"].astype(str)
    pred_intent = df["pred_intent"].fillna("unknown").astype(str)
    true_category = df["true_category"].astype(str)
    pred_category = df["pred_category"].fillna("unknown").astype(str)

    intent_precision, intent_recall, intent_f1, intent_support = (
        precision_recall_fscore_support(
            true_intent,
            pred_intent,
            labels=list(ALLOWED_INTENTS),
            zero_division=0,
        )
    )
    per_intent = [
        {
            "intent": intent,
            "precision": float(intent_precision[index]),
            "recall": float(intent_recall[index]),
            "f1": float(intent_f1[index]),
            "support": int(intent_support[index]),
        }
        for index, intent in enumerate(ALLOWED_INTENTS)
    ]

    errors = df.loc[
        true_intent.to_numpy() != pred_intent.to_numpy(),
        ["true_intent", "pred_intent"],
    ]
    confusion_rows = (
        errors.value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["count", "true_intent", "pred_intent"], ascending=[False, True, True])
        .head(10)
    )
    confusions = [
        {
            "true_intent": str(row.true_intent),
            "pred_intent": str(row.pred_intent),
            "count": int(row.count),
        }
        for row in confusion_rows.itertuples(index=False)
    ]

    confidence = pd.to_numeric(df["confidence"], errors="coerce").fillna(0).clip(0, 1)
    buckets = pd.cut(
        confidence,
        bins=[-1e-12, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        include_lowest=True,
    )
    calibration_frame = pd.DataFrame(
        {
            "bucket": buckets,
            "confidence": confidence,
            "correct": (true_intent.to_numpy() == pred_intent.to_numpy()).astype(float),
        }
    )
    calibration = []
    for bucket, group in calibration_frame.groupby("bucket", observed=False):
        calibration.append(
            {
                "bucket": str(bucket),
                "count": int(len(group)),
                "accuracy": float(group["correct"].mean()) if len(group) else None,
                "mean_confidence": float(group["confidence"].mean()) if len(group) else None,
            }
        )

    return {
        "run_id": run_id,
        "models": sorted(df["model"].dropna().astype(str).unique().tolist()),
        "count": int(len(df)),
        "intent_accuracy": float(accuracy_score(true_intent, pred_intent)),
        "category_accuracy": float(accuracy_score(true_category, pred_category)),
        "intent_macro_f1": float(
            f1_score(
                true_intent,
                pred_intent,
                labels=list(ALLOWED_INTENTS),
                average="macro",
                zero_division=0,
            )
        ),
        "category_macro_f1": float(
            f1_score(true_category, pred_category, average="macro", zero_division=0)
        ),
        "per_intent": per_intent,
        "top_confusions": confusions,
        "confidence_calibration": calibration,
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
        ["Metric", "Value"],
        [
            ("Tickets", report["count"]),
            ("Intent accuracy", report["intent_accuracy"]),
            ("Category accuracy", report["category_accuracy"]),
            ("Intent macro-F1", report["intent_macro_f1"]),
            ("Category macro-F1", report["category_macro_f1"]),
        ],
    )
    per_intent = _markdown_table(
        ["Intent", "Precision", "Recall", "F1", "Support"],
        (
            (row["intent"], row["precision"], row["recall"], row["f1"], row["support"])
            for row in report["per_intent"]
        ),
    )
    confusions = _markdown_table(
        ["True intent", "Predicted intent", "Count"],
        (
            (row["true_intent"], row["pred_intent"], row["count"])
            for row in report["top_confusions"]
        ),
    )
    calibration = _markdown_table(
        ["Confidence bucket", "Count", "Accuracy", "Mean confidence"],
        (
            (row["bucket"], row["count"], row["accuracy"], row["mean_confidence"])
            for row in report["confidence_calibration"]
        ),
    )
    models = ", ".join(report["models"]) or "unknown"
    return (
        f"# Classification report: {report['run_id']}\n\n"
        f"Models: {models}\n\n"
        f"## Overall\n\n{summary}\n\n"
        f"## Per-intent metrics\n\n{per_intent}\n\n"
        f"## Top 10 confusions\n\n{confusions}\n\n"
        f"## Accuracy by confidence bucket\n\n{calibration}\n"
    )


def write_report(
    report: dict[str, Any],
    output_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = report["run_id"]
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a filename-safe identifier")
    json_path = output_dir / f"{run_id}.json"
    markdown_path = output_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
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
