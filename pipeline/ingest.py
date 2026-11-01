"""Dataset ingestion and stratification pipeline.

Downloads the Bitext Customer Support dataset, performs a deterministic
stratified-by-intent 80/20 train/test split (seed 42), and bulk-inserts
records into the tickets table via COPY.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_FILE = (
    DEFAULT_DATA_DIR
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def download_dataset(
    url: str = DATA_URL,
    dest: Path = DEFAULT_DATA_FILE,
) -> Path:
    """Download dataset if not already cached locally."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached dataset at: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download completed.")
    return dest


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = "intent",
    test_size: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministically split dataframe into train and test sets stratified by stratify_col.

    The split is done on UNIQUE instruction texts, so every copy of the same
    sentence lands on the same side. Otherwise identical tickets could appear
    in both train and test and leak answers into evaluation.
    """
    if "instruction" in df.columns:
        uniq = df.drop_duplicates(subset="instruction")[["instruction", stratify_col]]
        uniq = _split_rows(uniq.reset_index(drop=True), stratify_col, test_size, seed)
        out = df.copy()
        out["split"] = out["instruction"].map(dict(zip(uniq["instruction"], uniq["split"])))
        return out
    return _split_rows(df, stratify_col, test_size, seed)


def _split_rows(df: pd.DataFrame, stratify_col: str, test_size: float, seed: int) -> pd.DataFrame:
    shuffled = df.sample(frac=1.0, random_state=seed)
    group_sizes = shuffled.groupby(stratify_col)[stratify_col].transform("count")
    rank = shuffled.groupby(stratify_col).cumcount()
    n_train = (group_sizes * (1.0 - test_size)).round().astype(int)
    shuffled["split"] = np.where(rank < n_train, "train", "test")
    return shuffled.sort_index()


def prepare_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and format the Bitext dataset with stratified train/test split."""
    df = pd.read_csv(csv_path)

    # Standardize column names
    if "response" in df.columns and "reference_response" not in df.columns:
        df = df.rename(columns={"response": "reference_response"})

    columns_needed = [
        "instruction",
        "category",
        "intent",
        "flags",
        "reference_response",
    ]
    for col in columns_needed:
        if col not in df.columns:
            df[col] = None

    # Apply deterministic 80/20 stratified split by intent
    df = stratified_split(df, stratify_col="intent", test_size=0.2, seed=42)

    return df[
        [
            "instruction",
            "category",
            "intent",
            "flags",
            "reference_response",
            "split",
        ]
    ]


def bulk_insert_tickets(df: pd.DataFrame, db_url: str | None = None) -> int:
    """Idempotently truncate and bulk insert tickets into PostgreSQL using COPY."""
    import psycopg

    if not db_url:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://support:support@localhost:5433/support",
        )

    print(f"Connecting to database to bulk insert {len(df)} tickets...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE tickets RESTART IDENTITY CASCADE;")
            with cur.copy(
                "COPY tickets (instruction, category, intent, flags, reference_response, split) "
                "FROM STDIN"
            ) as copy:
                for row in df.itertuples(index=False):
                    copy.write_row((
                        row.instruction,
                        row.category,
                        row.intent,
                        row.flags,
                        row.reference_response,
                        row.split,
                    ))
        conn.commit()
    print("Bulk insert completed.")
    return len(df)


def main() -> None:
    """Main execution flow for downloading, splitting, and ingesting tickets."""
    csv_path = DEFAULT_DATA_FILE
    if not csv_path.exists():
        try:
            csv_path = download_dataset()
        except Exception as exc:
            print(f"Dataset not available locally and download failed: {exc}")
            return

    df = prepare_dataset(csv_path)

    print("\n--- Row counts per split ---")
    print(df["split"].value_counts().to_string())

    print("\n--- Row counts per category ---")
    print(df["category"].value_counts().to_string())

    try:
        bulk_insert_tickets(df)
    except Exception as exc:
        print(f"Note: Database insertion skipped or failed ({exc}).")


if __name__ == "__main__":
    main()
