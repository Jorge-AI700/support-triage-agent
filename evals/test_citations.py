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
