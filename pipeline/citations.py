"""Deterministic citation verification for RAG generated replies.

Checks that all cited KB article IDs were actually retrieved, identifies any
sentences lacking an explicit citation, and evaluates overall grounding.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


def normalize_kb_id(val: str | int) -> str:
    """Normalize KB identifier to standard format 'KB-XX' (e.g. 1 -> 'KB-01', 'kb-7' -> 'KB-07')."""
    if isinstance(val, int):
        return f"KB-{val:02d}"
    s = str(val).strip().upper()
    match = re.match(r"^KB-?(\d+)$", s)
    if match:
        return f"KB-{int(match.group(1)):02d}"
    return s


def extract_citations(text: str) -> list[str]:
    """Extract all inline citations like '[KB-01]' in appearance order."""
    matches = re.findall(r"\[KB-(\d+)\]", text, flags=re.IGNORECASE)
    # Deduplicate while preserving order of appearance
    seen = set()
    cited = []
    for num in matches:
        kb_id = f"KB-{int(num):02d}"
        if kb_id not in seen:
            seen.add(kb_id)
            cited.append(kb_id)
    return cited


def split_sentences(text: str) -> list[str]:
    """Deterministically split response text into individual sentences.

    Handles paragraphs, bullet points, and numbered lists while avoiding false
    splits on list numbers (e.g. '1. ') or abbreviations.
    """
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip list markers like '1. ', '2. ', '- ', '* ' from the start of the line
        cleaned = re.sub(r"^(\d+\.|\*|-)\s*", "", line).strip()
        if not cleaned:
            continue

        # Split on sentence terminals (. ! ?) that are followed by a space and an uppercase letter/bracket
        # Use negative lookbehind (?<!\b\d) to avoid splitting on decimal numbers or digits
        raw_parts = re.split(r"(?<!\b\d)(?<=[.!?])\s+(?=[A-Z0-9\[])", cleaned)
        for part in raw_parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def check_citations(
    reply: str,
    retrieved_article_ids: Iterable[str | int],
) -> dict[str, Any]:
    """Verify that all citations in the reply are grounded in retrieved articles.

    Args:
        reply: The generated text response containing [KB-xx] citations.
        retrieved_article_ids: Collection of article IDs (e.g., ['KB-01', 'KB-07'] or [1, 7])
            that were retrieved from pgvector for this query.

    Returns:
        A dict with:
        - all_cited_ids_retrieved: True if at least one citation exists and all citations
          in the reply match an article in retrieved_article_ids.
        - uncited_sentences: List of sentences that do not contain a [KB-xx] citation.
        - grounded: True if all cited IDs were retrieved, at least one citation exists,
          and there are zero uncited sentences.
    """
    # 1. Normalize retrieved IDs into standard 'KB-XX' strings
    retrieved_set = {normalize_kb_id(aid) for aid in retrieved_article_ids}

    # 2. Extract citations present in the reply
    cited_ids = extract_citations(reply)

    # 3. Verify all cited IDs are within the retrieved set
    # False if no citations were made, or if any cited ID is outside retrieved_set
    all_cited_ids_retrieved = bool(cited_ids) and set(cited_ids).issubset(retrieved_set)

    # 4. Find any sentences that lack an inline [KB-xx] citation
    sentences = split_sentences(reply)
    uncited_sentences: list[str] = []
    citation_pattern = re.compile(r"\[KB-\d+\]", re.IGNORECASE)

    for sentence in sentences:
        if not citation_pattern.search(sentence):
            uncited_sentences.append(sentence)

    # 5. Determine overall grounding
    # Grounded requires:
    # - At least one valid citation
    # - All cited IDs exist in the retrieved set (no hallucinations)
    # - Every sentence has a citation (no unsupported factual claims)
    grounded = all_cited_ids_retrieved and len(uncited_sentences) == 0

    return {
        "all_cited_ids_retrieved": all_cited_ids_retrieved,
        "uncited_sentences": uncited_sentences,
        "grounded": grounded,
    }
