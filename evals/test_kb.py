"""Unit tests verifying knowledge base generation and chunking logic."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_kb import ARTICLES
from scripts.load_kb import parse_article, split_article_into_chunks

KB_DIR = PROJECT_ROOT / "kb"


def test_all_27_articles_exist():
    """Verify that all 27 intent articles are present with valid front matter."""
    assert len(ARTICLES) == 27
    files = sorted(KB_DIR.glob("*.md"))
    assert len(files) == 27, f"Expected 27 files, got {len(files)}"

    for idx, f in enumerate(files, start=1):
        content = f.read_text(encoding="utf-8")
        fm, title, body = parse_article(content)

        expected_id = f"KB-{idx:02d}"
        assert fm.get("id") == expected_id, f"{f.name}: expected id {expected_id}, got {fm.get('id')}"
        assert fm.get("intent") == f.stem, f"{f.name}: intent mismatch"
        assert "category" in fm and fm["category"], f"{f.name}: missing category"
        assert title and title != "Article", f"{f.name}: missing title"

        # Check required sections
        assert "## Steps" in body, f"{f.name}: missing Steps section"
        assert "## Frequently Asked Questions" in body, f"{f.name}: missing FAQ section"
        assert re.search(r"1\.\s+", body), f"{f.name}: missing numbered steps"
        assert "### " in body, f"{f.name}: missing FAQ questions"

        # Check for unresolved placeholders
        unresolved = re.findall(r"\{\{.*?\}\}", content)
        assert not unresolved, f"{f.name}: found unresolved placeholders: {unresolved}"


def test_chunking_constraints():
    """Verify that section chunking produces clean chunks in the ~300-500 char range."""
    files = sorted(KB_DIR.glob("*.md"))
    all_chunks = []
    for f in files:
        _, _, body = parse_article(f.read_text(encoding="utf-8"))
        chunks = split_article_into_chunks(body)
        assert len(chunks) >= 2, f"{f.name}: expected at least 2 chunks, got {len(chunks)}"
        all_chunks.extend(chunks)

    lens = [len(c) for c in all_chunks]
    avg_len = sum(lens) / len(lens)
    assert 250 <= avg_len <= 450, f"Average chunk length out of range: {avg_len}"
    assert max(lens) <= 600, f"Max chunk length too large: {max(lens)}"


if __name__ == "__main__":
    test_all_27_articles_exist()
    test_chunking_constraints()
    print("test_all_27_articles_exist: PASSED")
    print("test_chunking_constraints: PASSED")
