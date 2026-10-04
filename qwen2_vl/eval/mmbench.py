"""Generate MMBench predictions from TSV files."""

import argparse
import base64
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..inference import add_model_arguments, build_from_args, generate_answer
from .common import add_output_arguments, answer_record, chunk_examples, open_output, write_answer


def present(value):
    return not pd.isna(value) and str(value).strip().lower() not in ("", "none", "nan")


def decode_image(value, rows):
    seen = set()
    # Some MMBench rows refer to another row's image by index.
    while len(str(value)) < 64 and str(value) in rows:
        if str(value) in seen:
            raise ValueError("Cyclic MMBench image reference.")
        seen.add(str(value))
        value = rows[str(value)]["image"]
    with Image.open(BytesIO(base64.b64decode(str(value).split(",")[-1]))) as image:
        return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL MMBench inference.")
    add_model_arguments(parser)
    add_output_arguments(parser)
    parser.set_defaults(max_new_tokens=32)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--lang", choices=("en", "cn"), default="en")
    args = parser.parse_args()
    table = pd.read_table(args.question_file, dtype={"image": str}).to_dict("records")
    rows = {str(row["index"]): row for row in table}
    examples = chunk_examples(table, args.num_chunks, args.chunk_idx)
    model, processor = build_from_args(args)
    with open_output(args.answers_file) as stream:
        for row in tqdm(examples, desc="MMBench"):
            options = [str(row[letter]) for letter in "ABCD" if letter in row and present(row[letter])]
            if not options:
                raise ValueError(f"No options for question {row['index']}.")
            letters = list("ABCD"[:len(options)])
            image = decode_image(row["image"], rows)
            for round_idx in range(len(options) if args.all_rounds else 1):
                prompt = str(row["question"])
                if present(row.get("hint")):
                    prompt = str(row["hint"]) + "\n" + prompt
                prompt += "\n" + "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options))
                instruction = "请直接回答选项字母。" if args.lang == "cn" else "Answer with the option's letter from the given choices directly."
                prompt += "\n" + instruction
                answer = generate_answer(
                    model, processor, prompt, image=image,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                )
                record = answer_record(args, row["index"], prompt, answer)
                record.update(round_id=round_idx, options=options, option_char=letters)
                write_answer(stream, record)
                options = options[1:] + options[:1]
                letters = letters[1:] + letters[:1]


if __name__ == "__main__":
    main()
