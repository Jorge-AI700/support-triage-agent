"""Unit tests verifying train/test split determinism and stratification."""

import pandas as pd
from pipeline.ingest import stratified_split


def test_split_is_deterministic():
    """Verify that the split produces identical assignments across runs with the same seed."""
    df = pd.DataFrame({
        "intent": ["cancel_order"] * 50 + ["refund"] * 30 + ["change_address"] * 20,
        "instruction": [f"query {i}" for i in range(100)],
        "category": ["ORDER"] * 50 + ["REFUND"] * 30 + ["ACCOUNT"] * 20,
        "flags": ["tag"] * 100,
        "reference_response": ["resp"] * 100,
    })

    split1 = stratified_split(df, stratify_col="intent", test_size=0.2, seed=42)
    split2 = stratified_split(df, stratify_col="intent", test_size=0.2, seed=42)

    pd.testing.assert_series_equal(split1["split"], split2["split"])


def test_split_is_stratified():
    """Verify that each intent group preserves the 80/20 train/test ratio."""
    df = pd.DataFrame({
        "intent": ["cancel_order"] * 100 + ["refund"] * 50 + ["login_issue"] * 25,
        "instruction": [f"query {i}" for i in range(175)],
    })

    split = stratified_split(df, stratify_col="intent", test_size=0.2, seed=42)

    counts = split.groupby(["intent", "split"]).size().unstack(fill_value=0)
    assert counts.loc["cancel_order", "train"] == 80
    assert counts.loc["cancel_order", "test"] == 20
    assert counts.loc["refund", "train"] == 40
    assert counts.loc["refund", "test"] == 10
    assert counts.loc["login_issue", "train"] == 20
    assert counts.loc["login_issue", "test"] == 5

    total_counts = split["split"].value_counts().to_dict()
    assert total_counts["train"] == 140
    assert total_counts["test"] == 35


if __name__ == "__main__":
    test_split_is_deterministic()
    test_split_is_stratified()
    print("test_split_is_deterministic: PASSED")
    print("test_split_is_stratified: PASSED")
