"""Dataset loading helpers shared by the LoRA training scripts."""

import glob as glob_module
import json
import os
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset


INSTRUCTION_KEYS = ("instruction", "prompt", "question", "query", "task")
INPUT_KEYS = ("input", "context", "ctx", "data")
OUTPUT_KEYS = ("output", "answer", "response", "completion")
MESSAGE_KEYS = ("messages", "conversations")


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def _is_empty(value):
    return value in (None, "", {}, [])


def _first_value(batch, keys, index):
    for key in keys:
        values = batch.get(key)
        if values is None:
            continue
        value = values[index]
        if not _is_empty(value):
            return value
    return ""


def _extract_chat_pair(batch, index):
    messages = _first_value(batch, MESSAGE_KEYS, index)
    if not isinstance(messages, list):
        return "", ""

    user_parts = []
    assistant_parts = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(
            message.get("role")
            or message.get("from")
            or message.get("speaker")
            or ""
        ).lower()
        content = (
            message.get("content")
            or message.get("value")
            or message.get("text")
            or ""
        )

        if role in {"user", "human"}:
            user_parts.append(_to_text(content))
        elif role in {"assistant", "gpt", "model"}:
            assistant_parts.append(_to_text(content))

    return "\n\n".join(part for part in user_parts if part), "\n\n".join(
        part for part in assistant_parts if part
    )


def _normalizer(source_name):
    def normalize_batch(batch):
        if not batch:
            return {"instruction": [], "input": [], "output": [], "source_file": []}

        batch_size = len(next(iter(batch.values())))
        normalized = {
            "instruction": [],
            "input": [],
            "output": [],
            "source_file": [],
        }

        for index in range(batch_size):
            chat_instruction, chat_output = _extract_chat_pair(batch, index)
            instruction = _first_value(batch, INSTRUCTION_KEYS, index) or chat_instruction
            output = _first_value(batch, OUTPUT_KEYS, index) or chat_output
            input_data = _first_value(batch, INPUT_KEYS, index)

            normalized["instruction"].append(_to_text(instruction))
            normalized["input"].append(_to_text(input_data))
            normalized["output"].append(_to_text(output))
            normalized["source_file"].append(source_name)

        return normalized

    return normalize_batch


def _split_entries(value):
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _dataset_dir(script_dir):
    value = os.environ.get("DATASET_DIR")
    if value in (None, ""):
        return script_dir

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _expand_entry(entry, base_dir):
    entry_path = Path(entry).expanduser()
    candidates = [entry_path] if entry_path.is_absolute() else [Path.cwd() / entry, base_dir / entry]

    matches = []
    for candidate in candidates:
        pattern = str(candidate)
        if glob_module.has_magic(pattern):
            matches.extend(Path(match) for match in glob_module.glob(pattern, recursive=True))
        elif candidate.exists():
            matches.append(candidate)

        if matches:
            break

    if not matches:
        raise FileNotFoundError(
            f"Dataset path or glob did not match anything: {entry!r} "
            f"(searched from {Path.cwd()} and {base_dir})"
        )

    return sorted(path.resolve() for path in matches if path.is_file())


def _expand_entries(entries, base_dir):
    files = []
    seen = set()

    for entry in entries:
        for path in _expand_entry(entry, base_dir):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)

    return files


def _resolve_train_files(default_train_name, base_dir):
    if os.environ.get("TRAIN_FILES"):
        return _expand_entries(_split_entries(os.environ["TRAIN_FILES"]), base_dir), "TRAIN_FILES"
    if os.environ.get("DATASET_FILES"):
        return _expand_entries(_split_entries(os.environ["DATASET_FILES"]), base_dir), "DATASET_FILES"
    if os.environ.get("DATASET_GLOB"):
        return _expand_entries(_split_entries(os.environ["DATASET_GLOB"]), base_dir), "DATASET_GLOB"
    if os.environ.get("TRAIN_FILE"):
        return _expand_entries([os.environ["TRAIN_FILE"]], base_dir), "TRAIN_FILE"
    return _expand_entries([default_train_name], base_dir), "default"


def _resolve_validation_files(default_valid_name, base_dir, train_source):
    if os.environ.get("VALID_FILES"):
        return _expand_entries(_split_entries(os.environ["VALID_FILES"]), base_dir), "VALID_FILES"
    if os.environ.get("VALID_FILE"):
        return _expand_entries([os.environ["VALID_FILE"]], base_dir), "VALID_FILE"

    multi_source_mode = train_source in {"TRAIN_FILES", "DATASET_FILES", "DATASET_GLOB"}
    if multi_source_mode:
        return [], "auto_split"

    return _expand_entries([default_valid_name], base_dir), "default"


def _load_one_json(path):
    cache_dir = os.environ.get("DATASET_CACHE_DIR") or None
    dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
        cache_dir=cache_dir,
    )

    return dataset.map(
        _normalizer(path.name),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Normalize {path.name}",
        load_from_cache_file=True,
    )


def _load_many_json(files, split_name):
    datasets = []
    for path in files:
        dataset = _load_one_json(path)
        if len(dataset) == 0:
            print(f"Skipping empty {split_name} dataset: {path}")
            continue
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No non-empty {split_name} datasets were loaded.")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _limit_dataset(dataset, env_name, split_name):
    max_samples = env_int(env_name, 0)
    if max_samples <= 0:
        return dataset

    limit = min(max_samples, len(dataset))
    print(f"{split_name}: limiting to {limit}/{len(dataset)} examples via {env_name}.")
    return dataset.select(range(limit))


def load_instruction_datasets(script_dir, default_train_name, default_valid_name):
    """Load and normalize train/validation JSON datasets.

    Fast dataset workflows:
    - TRAIN_FILES / VALID_FILES: comma-separated paths or globs.
    - DATASET_FILES / DATASET_GLOB: train sources, with validation auto-split.
    - DATASET_DIR: base directory for relative dataset paths.
    """

    base_dir = _dataset_dir(script_dir)
    train_files, train_source = _resolve_train_files(default_train_name, base_dir)
    valid_files, valid_source = _resolve_validation_files(default_valid_name, base_dir, train_source)

    print("Dataset base directory:", base_dir)
    print(f"Train source ({train_source}):")
    for path in train_files:
        print(f"  - {path}")

    if valid_files:
        print(f"Validation source ({valid_source}):")
        for path in valid_files:
            print(f"  - {path}")
    else:
        print("Validation source: automatic split from training data")

    seed = env_int("DATASET_SEED", 42)
    shuffle = env_bool("SHUFFLE_DATASETS", True)
    validation_split = env_float("VALIDATION_SPLIT", 0.05)

    train_dataset = _load_many_json(train_files, "train")

    if valid_files:
        validation_dataset = _load_many_json(valid_files, "validation")
        if shuffle:
            train_dataset = train_dataset.shuffle(seed=seed)
    else:
        if len(train_dataset) < 2:
            raise RuntimeError("Automatic validation split needs at least 2 training examples.")
        if validation_split <= 0 or validation_split >= 1:
            raise RuntimeError("VALIDATION_SPLIT must be between 0 and 1 when no validation file is provided.")

        split_dataset = train_dataset.train_test_split(
            test_size=validation_split,
            seed=seed,
            shuffle=shuffle,
        )
        train_dataset = split_dataset["train"]
        validation_dataset = split_dataset["test"]
        print(f"Automatic validation split: {validation_split:.3f}")

    train_dataset = _limit_dataset(train_dataset, "MAX_TRAIN_SAMPLES", "train")
    validation_dataset = _limit_dataset(validation_dataset, "MAX_VALID_SAMPLES", "validation")

    print(f"Dataset sizes: train={len(train_dataset)}, validation={len(validation_dataset)}")
    return DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )
