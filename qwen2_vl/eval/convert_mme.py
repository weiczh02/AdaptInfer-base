"""Convert MME predictions to the official scoring format."""

import argparse
from collections import defaultdict
from pathlib import Path

from .common import read_jsonl


def normalize_question(question):
    question = question.replace("Answer the question using a single word or phrase.", "").strip()
    if not question.endswith("Please answer yes or no."):
        question += " Please answer yes or no."
    return " ".join(question.split())


def load_ground_truth(data_path):
    ground_truth = {}
    for category in Path(data_path).expanduser().iterdir():
        if not category.is_dir():
            continue
        question_dir = category / "questions_answers_YN"
        if not question_dir.is_dir():
            question_dir = category
        for path in question_dir.glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                question, answer = line.split("\t", 1)
                key = (category.name, path.name, normalize_question(question))
                ground_truth[key] = answer.strip()
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Convert MME predictions.")
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--mme-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ground_truth = load_ground_truth(args.mme_data_path)
    results = defaultdict(list)
    for item in read_jsonl(args.answers_file):
        parts = str(item["question_id"]).replace("\\", "/").split("/")
        category, file = parts[0], Path(parts[-1]).with_suffix(".txt").name
        question = normalize_question(item["prompt"])
        key = (category, file, question)
        if key not in ground_truth:
            raise ValueError(f"MME ground truth not found: {key}")
        answer = " ".join(item["text"].split())
        results[category].append("\t".join((file, question, ground_truth[key], answer)))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, lines in results.items():
        (output_dir / f"{category}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
