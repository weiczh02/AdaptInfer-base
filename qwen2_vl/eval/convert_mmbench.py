"""Export single-round MMBench predictions for submission."""

import argparse
from pathlib import Path

import pandas as pd

from .common import read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Export MMBench predictions.")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    table = pd.read_table(args.annotation_file)
    table = table.drop(columns=["hint", "category", "source", "image", "comment", "l2-category"], errors="ignore")
    predictions = read_jsonl(args.answers_file)
    if any(item.get("round_id", 0) != 0 for item in predictions):
        raise ValueError("This exporter expects single-round predictions; score option rotations separately.")
    by_id = {str(item["question_id"]): item["text"] for item in predictions}
    table["prediction"] = table["index"].astype(str).map(by_id)
    if table["prediction"].isna().any():
        raise ValueError("Predictions are missing for some MMBench questions.")
    path = Path(args.output_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, index=False, engine="openpyxl")


if __name__ == "__main__":
    main()
