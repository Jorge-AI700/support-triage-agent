"""Unit tests for pipeline.citations citation verification."""

import pytest
from pipeline.citations import check_citations, extract_citations, normalize_kb_id, split_sentences


def test_normalize_kb_id():
    assert normalize_kb_id(1) == "KB-01"
    assert normalize_kb_id(27) == "KB-27"
    assert normalize_kb_id("KB-01") == "KB-01"
    assert normalize_kb_id("kb-7") == "KB-07"
    assert normalize_kb_id("KB-15") == "KB-15"


def test_extract_citations():
    text = "Follow steps in [KB-01]. If needed, refer to [KB-04] and [KB-01]."
    citations = extract_citations(text)
    assert citations == ["KB-01", "KB-04"]


def test_split_sentences():
    text = (
        "1. Log in to your account [KB-01].\n"
        "2. Click Cancel Order [KB-01].\n\n"
        "Refunds are processed in 3-5 days [KB-01]. Contact us if delayed."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Log in to your account [KB-01]."
    assert sentences[1] == "Click Cancel Order [KB-01]."
    assert sentences[2] == "Refunds are processed in 3-5 days [KB-01]."
    assert sentences[3] == "Contact us if delayed."


def test_grounded_reply():
    reply = (
        "You can cancel your order from your Order History dashboard [KB-01]. "
        "Orders that have already shipped cannot be canceled online [KB-01]."
    )
    res = check_citations(reply, retrieved_article_ids=["KB-01", "KB-04"])

    assert res["all_cited_ids_retrieved"] is True
    assert res["uncited_sentences"] == []
    assert res["grounded"] is True


def test_unretrieved_citation_hallucination():
    reply = (
        "To modify your order, visit the details page [KB-02]. "
        "Special items cannot be modified [KB-99]."
    )
    # KB-99 was not in the retrieved articles
    res = check_citations(reply, retrieved_article_ids=["KB-01", "KB-02"])

    assert res["all_cited_ids_retrieved"] is False
    assert res["grounded"] is False


def test_uncited_sentences_flagged():
    reply = (
        "To cancel your order, go to Account Settings [KB-01]. "
        "Have a wonderful day and thank you for shopping with us!"
    )
    res = check_citations(reply, retrieved_article_ids=["KB-01"])

    assert res["all_cited_ids_retrieved"] is True
    assert res["uncited_sentences"] == ["Have a wonderful day and thank you for shopping with us!"]
    assert res["grounded"] is False


def test_no_citations_reply():
    reply = "I am escalating this ticket to a human support agent."
    res = check_citations(reply, retrieved_article_ids=["KB-01"])

    assert res["all_cited_ids_retrieved"] is False
    assert res["uncited_sentences"] == ["I am escalating this ticket to a human support agent."]
    assert res["grounded"] is False


def test_integer_and_string_retrieved_ids():
    reply = "Follow step one [KB-01] and step two [KB-07]."
    # Pass integers for retrieved IDs
    res = check_citations(reply, retrieved_article_ids=[1, 7])

    assert res["all_cited_ids_retrieved"] is True
    assert res["uncited_sentences"] == []
    assert res["grounded"] is True


def test_uncited_greeting_is_not_exempted():
    reply = "Hello!\nTrack your order from the order page [KB-01]."
    res = check_citations(reply, retrieved_article_ids=["KB-01"])

    assert res["uncited_sentences"] == ["Hello!"]
    assert res["grounded"] is False


def test_v2_requires_citation_at_sentence_end_while_v1_remains_available():
    reply = "According to [KB-01], track your order from the order page."

    assert check_citations(reply, ["KB-01"], prompt_version="v1")["grounded"] is True
    v2 = check_citations(reply, ["KB-01"], prompt_version="v2")
    assert v2["uncited_sentences"] == [reply]
    assert v2["grounded"] is False


def test_inline_numbered_list_markers_are_not_uncited_sentences():
    from pipeline.citations import check_citations, split_sentences

    reply = "1. Sign in at acme-store.example.com [KB-02]. 2. Open the order [KB-02]. 3. Click Save [KB-02]."
    assert split_sentences(reply) == [
        "Sign in at acme-store.example.com [KB-02].",
        "Open the order [KB-02].",
        "Click Save [KB-02].",
    ]
    for version in ("v1", "v2"):
        result = check_citations(reply, ["KB-02"], prompt_version=version)
        assert result["grounded"] is True
        assert result["uncited_sentences"] == []
