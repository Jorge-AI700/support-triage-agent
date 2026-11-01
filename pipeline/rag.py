"""RAG (Retrieval-Augmented Generation) pipeline for support ticket auto-replies.

Retrieves top-k knowledge base chunks using pgvector vector search, prompts the
LLM to generate a grounded response citing knowledge base articles ([KB-xx]),
and validates citations against the retrieved context.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pipeline.citations import check_citations, extract_citations
from pipeline.llm import OpenRouterClient
from pipeline.retrieve import (
    DEFAULT_DATABASE_URL,
    retrieve_kb_chunks,
    retrieve_similar_tickets,
)

SYSTEM_PROMPT = """You are an expert customer support agent for Acme Store.
Draft a clear, concise, and helpful reply to the customer's support request.

Rules you MUST strictly follow:
1. Answer ONLY using facts directly stated in the provided Knowledge Base context chunks below. Do not assume, extrapolate, or introduce outside information.
2. Every single sentence or list item in your reply MUST include an inline citation tag pointing to the article ID, formatted exactly as [KB-xx] (e.g. [KB-01]). Do NOT include any ungrounded introductory lines (such as 'To cancel your order, please follow these steps:') or conclusions that lack a citation.
3. If the provided Knowledge Base chunks do not contain sufficient information to answer the inquiry, state: 'The provided knowledge base does not contain information to resolve this issue. I will escalate your ticket to a human support agent.'
4. Be direct, concise, and begin immediately with the cited steps or facts."""

RAG_PROMPT_VERSION = "grounded-reply-v1"


class LLMClient(Protocol):
    model: str

    def complete(self, messages: Sequence[Mapping[str, str]]) -> Any: ...


class Cache(Protocol):
    def get(self, key: str) -> Mapping[str, Any] | None: ...

    def set(self, key: str, response: Mapping[str, Any]) -> None: ...


def build_user_prompt(
    ticket_text: str,
    predicted_intent: str | None,
    kb_chunks: Sequence[dict[str, Any]],
) -> str:
    """Format the ticket information and retrieved knowledge base context chunks for the LLM."""
    context_blocks = []
    for chunk in kb_chunks:
        kb_id = chunk.get("kb_id", f"KB-{chunk.get('article_id', 0):02d}")
        title = chunk.get("article_title", "Knowledge Base Article")
        text = chunk.get("chunk", "").strip()
        context_blocks.append(f"[{kb_id}] {title}\n{text}")

    context_str = "\n\n---\n\n".join(context_blocks)
    intent_str = predicted_intent if predicted_intent else "unspecified"

    return (
        f"Customer Ticket:\n{ticket_text}\n\n"
        f"Predicted Intent: {intent_str}\n\n"
        f"Knowledge Base Context Chunks:\n{context_str}\n\n"
        f"Please write the grounded reply following all citation rules:"
    )


def reply_cache_key(
    model: str,
    ticket_text: str,
    predicted_intent: str | None,
    kb_chunks: Sequence[dict[str, Any]],
) -> str:
    """Build a stable cache key from every input that can affect the reply."""
    user_prompt = build_user_prompt(ticket_text, predicted_intent, kb_chunks)
    raw = f"{model}{RAG_PROMPT_VERSION}{SYSTEM_PROMPT}{user_prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def draft_reply(
    ticket_text: str,
    predicted_intent: str | None = None,
    *,
    client: LLMClient | None = None,
    kb_k: int = 4,
    db_url: str | None = None,
    retrieved_chunks: Sequence[dict[str, Any]] | None = None,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Retrieve KB chunks, prompt the LLM for a grounded reply, and extract citations.

    Args:
        ticket_text: The user's support ticket instruction.
        predicted_intent: The classified intent (optional, assists context).
        client: Optional OpenRouterClient instance (constructed if None).
        kb_k: Number of knowledge base chunks to retrieve (default: 4).
        db_url: PostgreSQL connection string.
        retrieved_chunks: Optional already-retrieved chunks. Supplying these keeps
            retrieval and citation evaluation on the exact same evidence set.
        cache: Optional cache shared with classification and other LLM calls.

    Returns:
        dict with:
        - reply: The generated LLM response.
        - citations: List of cited KB IDs (e.g. ['KB-01']).
        - used_chunk_ids: IDs of retrieved chunks belonging to the cited articles.
        - retrieval_scores: Cosine similarity scores of retrieved chunks.
    """
    if client is None:
        client = OpenRouterClient()

    # 1. Retrieve top-k KB chunks unless the caller already did so.
    if retrieved_chunks is None:
        retrieved_chunks = retrieve_kb_chunks(ticket_text, k=kb_k, db_url=db_url)
    else:
        retrieved_chunks = list(retrieved_chunks)

    # 2. Build prompt and call LLM
    user_prompt = build_user_prompt(ticket_text, predicted_intent, retrieved_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    key = reply_cache_key(client.model, ticket_text, predicted_intent, retrieved_chunks)
    cached = cache.get(key) if cache is not None else None
    if cached is not None and isinstance(cached.get("reply"), str):
        reply = str(cached["reply"]).strip()
        usage: dict[str, Any] = {}
        cache_hit = True
    else:
        llm_resp = client.complete(messages)
        reply = llm_resp.content.strip()
        usage = dict(llm_resp.usage)
        cache_hit = False
        if cache is not None:
            cache.set(key, {"reply": reply})

    # 3. Extract citations and identify which chunk IDs correspond to cited articles
    citations = extract_citations(reply)
    cited_set = set(citations)

    used_chunk_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
        if chunk.get("kb_id") in cited_set
    ]

    retrieval_scores = [
        round(float(chunk["score"]), 4)
        for chunk in retrieved_chunks
    ]

    return {
        "reply": reply,
        "citations": citations,
        "used_chunk_ids": used_chunk_ids,
        "retrieval_scores": retrieval_scores,
        "usage": usage,
        "cache_hit": cache_hit,
        "cache_key": key,
    }


def run_rag_cli(ticket_id: int, db_url: str | None = None) -> None:
    """Run retrieval, drafting, and citation checking for a given ticket ID and print outputs."""
    db_url = db_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.id,
                    t.instruction,
                    t.intent,
                    t.category,
                    t.split,
                    p.pred_intent,
                    p.confidence
                FROM tickets t
                LEFT JOIN (
                    SELECT DISTINCT ON (ticket_id) ticket_id, pred_intent, confidence
                    FROM predictions
                    ORDER BY ticket_id, id DESC
                ) p ON t.id = p.ticket_id
                WHERE t.id = %s;
                """,
                (ticket_id,),
            )
            ticket = cur.fetchone()

    if not ticket:
        print(f"Error: Ticket #{ticket_id} not found in database.", file=sys.stderr)
        sys.exit(1)

    ticket_text = ticket["instruction"]
    intent = ticket["pred_intent"] or ticket["intent"]

    print("=" * 70)
    print(f"TICKET #{ticket['id']} (Split: {ticket['split']})")
    print(f"Instruction: {ticket_text}")
    print(f"Intent (GT): {ticket['intent']} | Predicted: {ticket['pred_intent']} (Conf: {ticket['confidence']})")
    print("=" * 70)

    # 1. Retrieval
    kb_chunks = retrieve_kb_chunks(ticket_text, k=4, db_url=db_url)
    similar_tickets = retrieve_similar_tickets(ticket_text, k=3, db_url=db_url)

    print("\n--- [RETRIEVAL: Top-4 Knowledge Base Chunks] ---")
    for idx, c in enumerate(kb_chunks, 1):
        preview = c["chunk"].replace("\n", " ")[:90]
        print(f"  {idx}. [{c['kb_id']}] {c['article_title']} (score: {c['score']:.4f}, chunk_id: {c['chunk_id']})")
        print(f"     Preview: {preview}...")

    print("\n--- [RETRIEVAL: Top-3 Similar Past Train Tickets] ---")
    for idx, t in enumerate(similar_tickets, 1):
        preview = t["instruction"].replace("\n", " ")[:90]
        print(f"  {idx}. Ticket #{t['ticket_id']} [{t['intent']}] (score: {t['score']:.4f})")
        print(f"     Instruction: {preview}...")

    # 2. Draft Reply
    rag_result = draft_reply(ticket_text, predicted_intent=intent, kb_k=4, db_url=db_url)
    reply = rag_result["reply"]

    print("\n--- [DRAFT REPLY] ---")
    print(reply)

    # 3. Citation Check
    retrieved_article_ids = [c["kb_id"] for c in kb_chunks]
    citation_check = check_citations(reply, retrieved_article_ids)

    print("\n--- [CITATION VERIFICATION] ---")
    print(f"  Grounded:                {citation_check['grounded']}")
    print(f"  All Cited IDs Retrieved: {citation_check['all_cited_ids_retrieved']}")
    print(f"  Extracted Citations:     {rag_result['citations']}")
    print(f"  Used Chunk IDs:          {rag_result['used_chunk_ids']}")
    print(f"  Retrieval Scores:        {rag_result['retrieval_scores']}")
    if citation_check["uncited_sentences"]:
        print(f"  Uncited Sentences ({len(citation_check['uncited_sentences'])}):")
        for sent in citation_check["uncited_sentences"]:
            print(f"    - {sent}")
    else:
        print("  Uncited Sentences:       None (100% cited)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG grounded auto-reply for a support ticket.")
    parser.add_argument("--ticket-id", type=int, required=True, help="ID of ticket from PostgreSQL tickets table.")
    parser.add_argument("--db-url", type=str, default=None, help="PostgreSQL connection string.")
    args = parser.parse_args()

    run_rag_cli(ticket_id=args.ticket_id, db_url=args.db_url)


if __name__ == "__main__":
    main()
