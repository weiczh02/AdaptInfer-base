"""Generate predictions for JSONL image question sets."""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..inference import add_model_arguments, build_from_args, generate_answer
from .common import add_output_arguments, answer_record, chunk_examples, open_output, read_jsonl, write_answer


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL image VQA inference.")
    add_model_arguments(parser)
    add_output_arguments(parser)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    args = parser.parse_args()
    questions = chunk_examples(read_jsonl(args.question_file), args.num_chunks, args.chunk_idx)
    model, processor = build_from_args(args)
    image_folder = Path(args.image_folder).expanduser()
    with open_output(args.answers_file) as stream:
        for item in tqdm(questions, desc="Image VQA"):
            with Image.open(image_folder / item["image"]) as source:
                image = source.convert("RGB")
            answer = generate_answer(
                model, processor, item["text"], image=image,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            write_answer(stream, answer_record(args, item["question_id"], item["text"], answer))


if __name__ == "__main__":
    main()
