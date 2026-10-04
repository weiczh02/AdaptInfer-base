import json
from pathlib import Path
from uuid import uuid4


def read_jsonl(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def chunk_examples(examples, num_chunks=1, chunk_idx=0):
    if num_chunks < 1 or not 0 <= chunk_idx < num_chunks:
        raise ValueError("Require num_chunks >= 1 and 0 <= chunk_idx < num_chunks.")
    size = (len(examples) + num_chunks - 1) // num_chunks
    return examples[chunk_idx * size:(chunk_idx + 1) * size]


def add_output_arguments(parser):
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)


def open_output(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_answer(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def answer_record(args, question_id, prompt, answer):
    return dict(
        question_id=question_id, prompt=prompt, text=answer, answer_id=uuid4().hex,
        model_id=Path(args.model_path.rstrip("/")).name,
        metadata={"pruning_layers": args.pruning_layers, "keep_ratios": args.keep_ratios},
    )
