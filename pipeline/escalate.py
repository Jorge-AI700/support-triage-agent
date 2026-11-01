"""Deterministic escalation rules for the support-ticket pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


# Keep policy values together so they can be reviewed and tuned without changing
# the rule flow. Scores are cosine similarities in the [0, 1] range used by the
# retriever: values below 0.50 are treated as weak matches.
ESCALATION_CONFIG: dict[str, Any] = {
    "confidence_threshold": 0.60,
    "kb_score_threshold": 0.50,
    "knn_agreement_threshold": 0.60,
    "human_required_intents": frozenset({"contact_human_agent", "complaint"}),
    # This is intentionally small and explicit. Word-boundary matching below
    # avoids false positives such as matching "hell" inside "hello".
    "angry_or_offensive_keywords": (
        "angry",
        "furious",
        "unacceptable",
        "hate",
        "idiot",
        "stupid",
        "bullshit",
        "fuck",
        "fucking",
    ),
}


def _contains_flagged_language(ticket_text: str) -> bool:
    """Return whether the ticket contains a configured whole word or phrase."""
    normalized = ticket_text.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", normalized)
        for keyword in ESCALATION_CONFIG["angry_or_offensive_keywords"]
    )


def should_escalate(
    confidence: float,
    citation_check: Mapping[str, Any],
    retrieval_scores: Sequence[float],
    ticket_text: str,
    predicted_intent: str,
    *,
    prompt_version: str = "v2",
    knn_majority_intent: str | None = None,
    knn_agreement: float | None = None,
) -> dict[str, Any]:
    """Evaluate all escalation rules and return stable, report-friendly reasons.

    Rules are additive: a ticket can expose several independent risks, so the
    caller receives every applicable reason rather than only the first one.
    """
    reasons: list[str] = []

    if prompt_version not in {"v1", "v2"}:
        raise ValueError("prompt_version must be 'v1' or 'v2'")

    if confidence < ESCALATION_CONFIG["confidence_threshold"]:
        reasons.append("low_confidence")

    if (
        prompt_version == "v2"
        and knn_majority_intent is not None
        and predicted_intent != knn_majority_intent
        and knn_agreement is not None
        and float(knn_agreement) >= ESCALATION_CONFIG["knn_agreement_threshold"]
    ):
        reasons.append("llm_knn_disagreement")

    # ``check_citations`` defines grounded as: at least one citation, every
    # citation belongs to the retrieved set, and every sentence is cited. The
    # detail checks keep this rule safe even if a caller supplies an inconsistent
    # hand-built citation result.
    has_uncited_sentences = bool(citation_check.get("uncited_sentences"))
    has_unverified_citations = citation_check.get("all_cited_ids_retrieved") is False
    if (
        citation_check.get("grounded") is not True
        or has_uncited_sentences
        or has_unverified_citations
    ):
        reasons.append("not_grounded")

    # No retrieval result is also weak retrieval: there is no KB evidence from
    # which an automated reply can safely be produced.
    top_kb_score = max((float(score) for score in retrieval_scores), default=None)
    if (
        top_kb_score is None
        or top_kb_score < ESCALATION_CONFIG["kb_score_threshold"]
    ):
        reasons.append("low_kb_score")

    if predicted_intent in ESCALATION_CONFIG["human_required_intents"]:
        reasons.append("human_required_intent")

    if _contains_flagged_language(ticket_text):
        reasons.append("angry_or_offensive_language")

    return {"escalate": bool(reasons), "reasons": reasons}
