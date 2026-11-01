"""Unit tests for every deterministic ticket-escalation rule."""

import pytest

from pipeline.escalate import ESCALATION_CONFIG, should_escalate


SAFE_CITATIONS = {
    "grounded": True,
    "uncited_sentences": [],
    "all_cited_ids_retrieved": True,
}


def decide(**overrides):
    inputs = {
        "confidence": 0.90,
        "citation_check": SAFE_CITATIONS,
        "retrieval_scores": [0.82, 0.71],
        "ticket_text": "Please tell me where my order is.",
        "predicted_intent": "track_order",
    }
    inputs.update(overrides)
    return should_escalate(**inputs)


def test_safe_ticket_is_not_escalated():
    assert decide() == {"escalate": False, "reasons": []}


def test_confidence_below_threshold_escalates():
    threshold = ESCALATION_CONFIG["confidence_threshold"]

    assert decide(confidence=threshold)["escalate"] is False
    assert decide(confidence=threshold - 0.01)["reasons"] == ["low_confidence"]


def test_v2_llm_knn_disagreement_at_threshold_escalates():
    threshold = ESCALATION_CONFIG["knn_agreement_threshold"]

    assert decide(
        knn_majority_intent="delivery_period",
        knn_agreement=threshold,
    )["reasons"] == ["llm_knn_disagreement"]
    assert decide(
        knn_majority_intent="delivery_period",
        knn_agreement=threshold - 0.01,
    )["escalate"] is False


def test_v1_rules_ignore_knn_disagreement():
    assert decide(
        prompt_version="v1",
        knn_majority_intent="delivery_period",
        knn_agreement=1.0,
    )["escalate"] is False


def test_knn_agreement_does_not_escalate_when_llm_agrees():
    assert decide(
        knn_majority_intent="track_order",
        knn_agreement=1.0,
    )["escalate"] is False


@pytest.mark.parametrize(
    "citation_check",
    [
        {
            "grounded": False,
            "uncited_sentences": ["This sentence has no source."],
            "all_cited_ids_retrieved": True,
        },
        {
            "grounded": False,
            "uncited_sentences": [],
            "all_cited_ids_retrieved": False,
        },
    ],
)
def test_ungrounded_reply_escalates(citation_check):
    assert decide(citation_check=citation_check)["reasons"] == ["not_grounded"]


def test_low_or_missing_kb_score_escalates():
    threshold = ESCALATION_CONFIG["kb_score_threshold"]

    assert decide(retrieval_scores=[threshold])["escalate"] is False
    assert decide(retrieval_scores=[threshold - 0.01])["reasons"] == ["low_kb_score"]
    assert decide(retrieval_scores=[])["reasons"] == ["low_kb_score"]


@pytest.mark.parametrize("intent", ["contact_human_agent", "complaint"])
def test_human_required_intent_escalates(intent):
    assert decide(predicted_intent=intent)["reasons"] == ["human_required_intent"]


@pytest.mark.parametrize(
    "ticket_text",
    [
        "I am FURIOUS about this delay.",
        "This is unacceptable.",
        "Only an idiot designed this checkout.",
    ],
)
def test_angry_or_offensive_language_escalates(ticket_text):
    assert decide(ticket_text=ticket_text)["reasons"] == [
        "angry_or_offensive_language"
    ]


def test_keywords_use_word_boundaries():
    assert decide(ticket_text="Hello, can you help?")["escalate"] is False


def test_all_applicable_reasons_are_returned_in_rule_order():
    result = decide(
        confidence=0.2,
        citation_check={
            "grounded": False,
            "uncited_sentences": ["No source."],
            "all_cited_ids_retrieved": False,
        },
        retrieval_scores=[],
        ticket_text="I am angry, this is bullshit.",
        predicted_intent="complaint",
    )

    assert result == {
        "escalate": True,
        "reasons": [
            "low_confidence",
            "not_grounded",
            "low_kb_score",
            "human_required_intent",
            "angry_or_offensive_language",
        ],
    }
