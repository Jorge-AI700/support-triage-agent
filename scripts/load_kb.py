"""Load knowledge base markdown articles into PostgreSQL and chunk them into sections.

Reads articles from kb/*.md, populates kb_articles, and splits each article into
clean section-based chunks (~300-500 chars) into kb_chunks with embedding left NULL.
"""

import os
import re
import sys
from pathlib import Path
import psycopg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KB_DIR = PROJECT_ROOT / "kb"


def parse_article(md_text: str) -> tuple[dict[str, str], str, str]:
    """Parse YAML front matter, title, and body from markdown document."""
    front_matter: dict[str, str] = {}
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_text, flags=re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                front_matter[key.strip()] = val.strip()
        body_md = md_text[fm_match.end() :].strip()
    else:
        body_md = md_text.strip()

    title_match = re.search(r"^#\s+(.+)$", body_md, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else front_matter.get("intent", "Article")

    return front_matter, title, body_md


def split_article_into_chunks(body_md: str) -> list[str]:
    """Split article into section-aligned chunks (~300-500 chars)."""
    # Split by level-2 markdown headings
    sections = re.split(r"\n(?=##\s)", body_md)
    chunks: list[str] = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # If a steps section exceeds ~550 chars, split cleanly into two sub-parts
        if len(sec) > 550 and "\n1." in sec:
            lines = sec.splitlines()
            header = lines[0]
            step_lines = [l for l in lines[1:] if l.strip()]
            mid = (len(step_lines) + 1) // 2
            c1 = header + "\n\n" + "\n".join(step_lines[:mid])
            c2 = header + " (Continued)\n\n" + "\n".join(step_lines[mid:])
            chunks.append(c1)
            chunks.append(c2)
        else:
            chunks.append(sec)

    return chunks


def load_knowledge_base(db_url: str | None = None) -> tuple[int, int]:
    """Load articles from kb/ into kb_articles and chunk into kb_chunks."""
    if not db_url:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://support:support@localhost:5433/support",
        )

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown articles found in {KB_DIR}. Run build_kb.py first.")

    total_articles = 0
    total_chunks = 0
    chunk_lengths: list[int] = []

    print(f"Connecting to database to load {len(md_files)} articles...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Idempotently reset kb_articles (cascades to kb_chunks)
            cur.execute("TRUNCATE TABLE kb_articles RESTART IDENTITY CASCADE;")

            for md_file in md_files:
                text = md_file.read_text(encoding="utf-8")
                fm, title, body_md = parse_article(text)
                intent = fm.get("intent", md_file.stem)

                cur.execute(
                    "INSERT INTO kb_articles (intent, title, body_md) "
                    "VALUES (%s, %s, %s) RETURNING id;",
                    (intent, title, body_md),
                )
                article_id = cur.fetchone()[0]
                total_articles += 1

                chunks = split_article_into_chunks(body_md)
                for chunk in chunks:
                    cur.execute(
                        "INSERT INTO kb_chunks (article_id, chunk, embedding) "
                        "VALUES (%s, %s, NULL);",
                        (article_id, chunk),
                    )
                    total_chunks += 1
                    chunk_lengths.append(len(chunk))

        conn.commit()

    avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"Loaded {total_articles} articles into kb_articles.")
    print(f"Created {total_chunks} chunks in kb_chunks.")
    print(f"Chunk size stats: min={min(chunk_lengths)} chars, max={max(chunk_lengths)} chars, avg={avg_len:.1f} chars.")

    return total_articles, total_chunks


def main() -> None:
    load_knowledge_base()


if __name__ == "__main__":
    main()
