"""Score TextVQA predictions with the shared benchmark metric."""

import argparse
import importlib.util
import json
import re
from pathlib import Path

from .common import read_jsonl


def extract_question(prompt):
    if prompt.startswith("OCR tokens: "):
        match = re.search(r"Question: (.*?) Short answer:", prompt, re.DOTALL)
        if match is None:
            raise ValueError("Question not found in the OCR prompt.")
        return match.group(1).lower()
    lines = prompt.split("\n")
    if prompt.startswith("Reference OCR token:") and len(lines) >= 2:
        return lines[1].lower()
    return lines[0].lower()


def load_metric():
    # Load the shared metric without importing the LLaVA model or its dependencies.
    path = Path(__file__).resolve().parents[2] / "llava" / "eval" / "m4c_evaluator.py"
    spec = importlib.util.spec_from_file_location("adaptinfer_textvqa_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


def main():
    parser = argparse.ArgumentParser(description="Score TextVQA predictions.")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    with Path(args.annotation_file).expanduser().open(encoding="utf-8") as stream:
        annotations = json.load(stream)["data"]
    annotations = {(str(item["image_id"]), item["question"].lower()): item for item in annotations}
    predictions = []
    for item in read_jsonl(args.result_file):
        annotation = annotations[(str(item["question_id"]), extract_question(item["prompt"]))]
        predictions.append(dict(pred_answer=item["text"], gt_answers=annotation["answers"]))
    if not predictions:
        raise ValueError("The result file is empty.")
    accuracy = load_metric().eval_pred_list(predictions)
    print(f"Samples: {len(predictions)}\nAccuracy: {100 * accuracy:.2f}%")


if __name__ == "__main__":
    main()
