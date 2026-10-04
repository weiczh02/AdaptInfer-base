"""Generate predictions for TGIF, MSRVTT, and MSVD question sets."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ..inference import add_model_arguments, build_from_args, generate_answer
from .common import add_output_arguments, chunk_examples, open_output, write_answer


def pair_questions_and_answers(questions, answers):
    if all("question_id" in item for item in answers):
        by_id = {str(item["question_id"]): item for item in answers}
        if len(by_id) != len(answers):
            raise ValueError("Duplicate question IDs in the answer file.")
        return [(question, by_id[str(question["question_id"])]["answer"]) for question in questions]
    if len(questions) != len(answers):
        raise ValueError("Question and answer files must have the same length when answers have no IDs.")
    return [(question, answer["answer"]) for question, answer in zip(questions, answers)]


def find_video(video_dir, name):
    path = video_dir / str(name)
    if path.is_file():
        return path
    for extension in (".mp4", ".avi", ".mov", ".mkv", ".gif"):
        candidate = video_dir / f"{name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL video QA inference.")
    add_model_arguments(parser)
    add_output_arguments(parser)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=1000, help="Number of questions; 0 uses the full set.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    with Path(args.question_file).expanduser().open(encoding="utf-8") as stream:
        questions = json.load(stream)
    with Path(args.answer_file).expanduser().open(encoding="utf-8") as stream:
        answers = json.load(stream)
    examples = pair_questions_and_answers(questions, answers)
    if args.limit:
        examples = examples[:args.limit]
    # Pair ground truth before splitting so every shard keeps the correct answer.
    examples = chunk_examples(examples, args.num_chunks, args.chunk_idx)
    model, processor = build_from_args(args)
    video_dir = Path(args.video_dir).expanduser()
    with open_output(args.answers_file) as stream:
        for item, answer in tqdm(examples, desc="Video QA"):
            video = find_video(video_dir, item["video_name"])
            prediction = generate_answer(
                model, processor, item["question"], video=str(video), fps=args.fps,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            write_answer(stream, dict(id=item["question_id"], question=item["question"], answer=answer, pred=prediction))


if __name__ == "__main__":
    main()
