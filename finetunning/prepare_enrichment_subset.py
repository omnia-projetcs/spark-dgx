#!/usr/bin/env python3
"""Prepare smaller JSON chunks before running expensive dataset enrichment."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample and split a JSON instruction dataset before enrichment. "
            "Input and output are JSON arrays."
        )
    )
    parser.add_argument("--input", required=True, help="Source JSON dataset.")
    parser.add_argument("--output", required=True, help="Output JSON file or chunk prefix.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Maximum number of examples to keep. Use 0 for all examples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--mode",
        choices=("random", "head", "stride"),
        default="random",
        help="Sampling strategy.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Examples per chunk. Use 0 to write one file.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicates based on instruction/input/output text.",
    )
    return parser.parse_args()


def load_json_array(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def stable_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def example_key(example):
    if not isinstance(example, dict):
        return hashlib.sha256(stable_text(example).encode("utf-8")).hexdigest()

    parts = [
        stable_text(example.get("instruction") or example.get("prompt") or example.get("question")),
        stable_text(example.get("input") or example.get("context") or example.get("data")),
        stable_text(example.get("output") or example.get("answer") or example.get("response")),
    ]
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()


def dedupe_examples(examples):
    seen = set()
    unique = []
    for example in examples:
        key = example_key(example)
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def select_examples(examples, limit, mode, seed):
    if limit <= 0 or limit >= len(examples):
        selected = list(examples)
    elif mode == "head":
        selected = list(examples[:limit])
    elif mode == "stride":
        step = len(examples) / limit
        selected = [examples[int(index * step)] for index in range(limit)]
    else:
        rng = random.Random(seed)
        selected = rng.sample(examples, limit)

    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(selected)
    return selected


def chunk_path(output, part_index, part_count):
    suffix = output.suffix or ".json"
    stem = output.with_suffix("")
    width = max(3, len(str(part_count)))
    return stem.parent / f"{stem.name}.part{part_index:0{width}d}{suffix}"


def write_json(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    examples = load_json_array(input_path)
    original_count = len(examples)

    if args.dedupe:
        examples = dedupe_examples(examples)

    selected = select_examples(examples, args.limit, args.mode, args.seed)

    if args.chunk_size <= 0 or args.chunk_size >= len(selected):
        write_json(output_path, selected)
        written = [output_path]
    else:
        part_count = math.ceil(len(selected) / args.chunk_size)
        written = []
        for index in range(part_count):
            start = index * args.chunk_size
            end = start + args.chunk_size
            path = chunk_path(output_path, index + 1, part_count)
            write_json(path, selected[start:end])
            written.append(path)

    print("Enrichment subset prepared")
    print(f"Input: {input_path}")
    print(f"Original examples: {original_count:,}")
    if args.dedupe:
        print(f"After dedupe: {len(examples):,}")
    print(f"Selected examples: {len(selected):,}")
    print(f"Mode: {args.mode}, seed: {args.seed}")
    print(f"Files written: {len(written)}")
    for path in written[:10]:
        print(f"  - {path}")
    if len(written) > 10:
        print(f"  ... {len(written) - 10} more")


if __name__ == "__main__":
    main()
