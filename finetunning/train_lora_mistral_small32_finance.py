# train_lora_mistral_small32_finance.py
#
# Recommended installation:
# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install -U torch transformers datasets accelerate peft bitsandbytes sentencepiece protobuf mistral-common
#
# Launch:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TOKENIZERS_PARALLELISM=false
# python train_lora_mistral_small32_finance.py

import gc
import glob
import inspect
import os
import json
import math
from pathlib import Path

import torch
import transformers

from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from train_dataset_utils import load_instruction_datasets

try:
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except ImportError:
    ChatCompletionRequest = None
    MistralTokenizer = None


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


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


MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
MODEL_CLASS = os.environ.get("MODEL_CLASS", "Mistral3ForConditionalGeneration")

DEFAULT_TRAIN_FILE = "dataset_finance_qa_enriched.json"
DEFAULT_VALID_FILE = "dataset_finance_qa.json"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/finance-mistral-small3.2-lora")
FINAL_DIR = os.environ.get("FINAL_DIR", f"{OUTPUT_DIR}/final")

# Defaults optimized for Mistral-Small 3.2 (24B)
MAX_LENGTH = env_int("MAX_LENGTH", 512)
EFFECTIVE_BATCH_SIZE = env_int("EFFECTIVE_BATCH_SIZE", 16)
PER_DEVICE_TRAIN_BATCH_SIZE = env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
PER_DEVICE_EVAL_BATCH_SIZE = env_int(
    "PER_DEVICE_EVAL_BATCH_SIZE",
    PER_DEVICE_TRAIN_BATCH_SIZE,
)
GRADIENT_ACCUMULATION_STEPS = env_int(
    "GRADIENT_ACCUMULATION_STEPS",
    max(1, math.ceil(EFFECTIVE_BATCH_SIZE / PER_DEVICE_TRAIN_BATCH_SIZE)),
)

NUM_TRAIN_EPOCHS = env_float("NUM_TRAIN_EPOCHS", 1.0)
MAX_STEPS = env_int("MAX_STEPS", -1)
LEARNING_RATE = env_float("LEARNING_RATE", 1e-4)
WARMUP_RATIO = env_float("WARMUP_RATIO", 0.03)
WARMUP_STEPS = env_int("WARMUP_STEPS", 0)
LR_SCHEDULER_TYPE = os.environ.get("LR_SCHEDULER_TYPE", "cosine")

LORA_R = env_int("LORA_R", 8)
LORA_ALPHA = env_int("LORA_ALPHA", 16)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.05)

TOKENIZE_BATCH_SIZE = env_int("TOKENIZE_BATCH_SIZE", 256)
TOKENIZE_NUM_PROC = env_int("TOKENIZE_NUM_PROC", min(4, os.cpu_count() or 1))
GROUP_BY_LENGTH = env_bool("GROUP_BY_LENGTH", True)
PAD_TO_MULTIPLE_OF = env_int("PAD_TO_MULTIPLE_OF", 8)
DATALOADER_NUM_WORKERS = env_int("DATALOADER_NUM_WORKERS", min(4, os.cpu_count() or 1))

# Default gradient checkpointing to true since 24B is memory intensive
GRADIENT_CHECKPOINTING = env_bool("GRADIENT_CHECKPOINTING", True)
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "auto")
OPTIM = os.environ.get("OPTIM", "paged_adamw_8bit")
TORCH_COMPILE = env_bool("TORCH_COMPILE", False)
AUTO_RESUME = env_bool("AUTO_RESUME", True)

LOGGING_STEPS = env_int("LOGGING_STEPS", 50)
SAVE_STEPS = env_int("SAVE_STEPS", 1000)
EVAL_STEPS = env_int("EVAL_STEPS", 1000)
SAVE_TOTAL_LIMIT = env_int("SAVE_TOTAL_LIMIT", 3)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# SYSTEM PROMPT - FINANCE
# ============================================================

SYSTEM_PROMPT = """You are an expert finance and market analysis assistant.

Your role:
- analyze companies, markets, financial statements, macroeconomic context, risks, and valuation drivers;
- explain trading, portfolio, accounting, and corporate finance concepts;
- produce structured reports, calculations, and decision-support summaries;
- distinguish facts, assumptions, estimates, and uncertainty;
- help with risk management, compliance-aware analysis, and due diligence.

Important rules:
- do not invent prices, filings, ratios, ratings, dates, or sources;
- if information is insufficient, say so clearly;
- this is educational analysis, not personalized financial advice;
- prioritize structured, traceable reasoning;
- if the expected format is JSON, produce only valid JSON.
"""


# ============================================================
# TOKENIZER
# ============================================================

if MistralTokenizer is None or ChatCompletionRequest is None:
    raise RuntimeError(
        "Mistral-Small-3.2 requires mistral-common for tokenization. "
        "Install it with: pip install -U mistral-common"
    )

tokenizer = MistralTokenizer.from_hf_hub(MODEL_NAME)


def nested_attr(obj, path):
    current = obj
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


EOS_TOKEN_ID = (
    getattr(tokenizer, "eos_id", None)
    or nested_attr(tokenizer, "instruct_tokenizer.tokenizer.eos_id")
)
PAD_TOKEN_ID = (
    getattr(tokenizer, "pad_id", None)
    or nested_attr(tokenizer, "instruct_tokenizer.tokenizer.pad_id")
    or EOS_TOKEN_ID
)

if PAD_TOKEN_ID is None:
    raise RuntimeError("Unable to resolve a pad/eos token id from the Mistral tokenizer.")


# ============================================================
# DATASET
# ============================================================

raw_dataset = load_instruction_datasets(
    SCRIPT_DIR,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VALID_FILE,
)

print(raw_dataset)
print("Train columns:", raw_dataset["train"].column_names)

tokenize_num_proc = max(
    1,
    min(TOKENIZE_NUM_PROC, *(len(raw_dataset[split]) for split in raw_dataset.keys())),
)
if tokenize_num_proc != TOKENIZE_NUM_PROC:
    print(f"TOKENIZE_NUM_PROC adjusted to {tokenize_num_proc} for dataset size.")


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_user_content(example):
    instruction = to_text(example.get("instruction", ""))
    input_data = example.get("input", "")

    parts = [
        "Instruction:",
        instruction,
    ]

    if input_data not in ("", None, {}, []):
        parts.extend([
            "\nContext / input data:",
            to_text(input_data),
        ])

    parts.append(
        "\nRespond as a finance and market analysis expert. "
        "Respect the format expected by the training example. "
        "If the expected output is JSON, produce only valid JSON."
    )

    return "\n".join(parts)


def build_messages(example, include_answer=True):
    user_text = build_user_content(example)
    assistant_text = to_text(example.get("output", ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    if include_answer:
        messages.append({"role": "assistant", "content": assistant_text})

    return messages


def encode_messages(messages):
    request = ChatCompletionRequest(messages=messages)
    return tokenizer.encode_chat_completion(request).tokens


def batch_to_examples(batch):
    batch_size = len(next(iter(batch.values())))
    for index in range(batch_size):
        yield {key: values[index] for key, values in batch.items()}


def tokenize_batch(batch):
    prompt_texts = []
    full_texts = []

    for example in batch_to_examples(batch):
        prompt_messages = build_messages(example, include_answer=False)
        full_messages = build_messages(example, include_answer=True)

        prompt_texts.append(encode_messages(prompt_messages)[:MAX_LENGTH])
        full_texts.append(encode_messages(full_messages))

    labels = []
    input_ids_list = []
    attention_masks = []
    was_truncated = []
    lengths = []

    for index, full_input_ids in enumerate(full_texts):
        input_ids = full_input_ids[:MAX_LENGTH]
        example_labels = input_ids.copy()
        prompt_len = len(prompt_texts[index])
        prompt_len = min(prompt_len, len(example_labels))
        example_labels[:prompt_len] = [-100] * prompt_len

        input_ids_list.append(input_ids)
        attention_masks.append([1] * len(input_ids))
        labels.append(example_labels)
        was_truncated.append(len(full_input_ids) > MAX_LENGTH)
        lengths.append(len(input_ids))

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels,
        "was_truncated": was_truncated,
        "length": lengths,
    }


columns_to_remove = raw_dataset["train"].column_names

tokenized_dataset = raw_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=TOKENIZE_BATCH_SIZE,
    num_proc=tokenize_num_proc,
    remove_columns=columns_to_remove,
    desc="Tokenization",
    load_from_cache_file=True,
)

print(tokenized_dataset)

for split in ["train", "validation"]:
    truncated_count = sum(tokenized_dataset[split]["was_truncated"])
    total = len(tokenized_dataset[split])
    print(f"{split}: {truncated_count}/{total} truncated examples")


# Remove examples where the assistant response is completely truncated.
def has_trainable_labels(example):
    return any(label != -100 for label in example["labels"])


before_train = len(tokenized_dataset["train"])
before_valid = len(tokenized_dataset["validation"])

tokenized_dataset = tokenized_dataset.filter(
    has_trainable_labels,
    num_proc=tokenize_num_proc,
    desc="Filtering examples without trainable labels",
)

print(f"Train kept: {len(tokenized_dataset['train'])}/{before_train}")
print(f"Validation kept: {len(tokenized_dataset['validation'])}/{before_valid}")

if len(tokenized_dataset["train"]) == 0:
    raise RuntimeError(
        "No valid training examples. "
        "MAX_LENGTH is probably too small or outputs are empty."
    )


# ============================================================
# DATA COLLATOR
# ============================================================

class DataCollatorForCausalLM:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(label + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


data_collator = DataCollatorForCausalLM(
    PAD_TOKEN_ID,
    pad_to_multiple_of=PAD_TO_MULTIPLE_OF if PAD_TO_MULTIPLE_OF > 0 else None,
)


# ============================================================
# CUDA / DTYPE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this QLoRA script.")

use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
cuda_major, cuda_minor = torch.cuda.get_device_capability(0)
use_tf32 = env_bool("TF32", cuda_major >= 8)

if use_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA capability: {cuda_major}.{cuda_minor}")
print(f"bf16 supported: {use_bf16}")
print(f"dtype used: {compute_dtype}")
print(f"TF32 enabled: {use_tf32}")
print(
    "Training profile: "
    f"max_length={MAX_LENGTH}, "
    f"micro_batch={PER_DEVICE_TRAIN_BATCH_SIZE}, "
    f"grad_accum={GRADIENT_ACCUMULATION_STEPS}, "
    f"effective_batch~={PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}, "
    f"gradient_checkpointing={GRADIENT_CHECKPOINTING}"
)


# ============================================================
# QUANTIZATION 4-BIT
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


def load_quantized_model():
    model_class = getattr(transformers, MODEL_CLASS, None)
    if model_class is None:
        raise RuntimeError(
            f"transformers.{MODEL_CLASS} is unavailable. "
            "Install a Transformers version that supports Mistral-Small-3.2."
        )

    base_kwargs = {
        "quantization_config": bnb_config,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }

    if ATTN_IMPLEMENTATION == "auto":
        candidates = ["flash_attention_2", "sdpa"]
    elif ATTN_IMPLEMENTATION in {"", "none", "default"}:
        candidates = [None]
    else:
        candidates = [ATTN_IMPLEMENTATION]

    last_error = None
    for attn_implementation in candidates:
        kwargs = dict(base_kwargs)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        try:
            model = model_class.from_pretrained(MODEL_NAME, **kwargs)
            print(f"Attention implementation: {attn_implementation or 'transformers default'}")
            return model
        except Exception as exc:
            last_error = exc
            if ATTN_IMPLEMENTATION != "auto":
                raise
            print(f"Attention implementation {attn_implementation} unavailable: {exc}")

    raise RuntimeError("Unable to load the model with any attention implementation.") from last_error


# ============================================================
# MODEL
# ============================================================

gc.collect()
torch.cuda.empty_cache()

model = load_quantized_model()

model.config.use_cache = False

memory_gb = model.get_memory_footprint() / 1024**3
print(f"Memory footprint after loading: {memory_gb:.2f} GB")

# 24B parameter model in 4-bit should be around 14-18 GB. Allow up to 35 GB.
if memory_gb > 35:
    raise RuntimeError(
        f"The model already occupies {memory_gb:.2f} GB. "
        "4-bit quantization is probably not applied correctly, "
        "or another model is loaded."
    )

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)


# ============================================================
# LORA CONFIG
# ============================================================

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

def build_training_arguments():
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "logging_steps": LOGGING_STEPS,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "eval_strategy": "steps",
        "eval_steps": EVAL_STEPS,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "tf32": use_tf32,
        "optim": OPTIM,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "group_by_length": GROUP_BY_LENGTH,
        "length_column_name": "length",
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": DATALOADER_NUM_WORKERS > 0,
        "torch_compile": TORCH_COMPILE,
        "include_tokens_per_second": True,
        "save_safetensors": True,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    if WARMUP_STEPS > 0:
        kwargs["warmup_steps"] = WARMUP_STEPS
    else:
        kwargs["warmup_ratio"] = WARMUP_RATIO

    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)

    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs) - set(filtered_kwargs))
    if ignored:
        print(f"Ignored TrainingArguments unsupported by this transformers version: {ignored}")

    return TrainingArguments(**filtered_kwargs)


training_args = build_training_arguments()


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)


# ============================================================
# TRAIN
# ============================================================

# Check if there are checkpoints in the output directory
resume_from_checkpoint = None
if AUTO_RESUME and os.path.exists(OUTPUT_DIR):
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if checkpoints:
        # Sort checkpoints by step number to find the latest
        def get_step_num(path):
            try:
                return int(path.split("-")[-1])
            except ValueError:
                return 0
        checkpoints.sort(key=get_step_num)
        resume_from_checkpoint = checkpoints[-1]
        print(f"Existing checkpoints found in {OUTPUT_DIR}. Resuming from: {resume_from_checkpoint}")
    else:
        print("No checkpoints found. Starting training from scratch.")
elif not AUTO_RESUME:
    print("AUTO_RESUME=false. Starting training from scratch.")
else:
    print("Output directory does not exist. Starting training from scratch.")

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.save_model(FINAL_DIR)
if hasattr(tokenizer, "save_pretrained"):
    tokenizer.save_pretrained(FINAL_DIR)
else:
    print("Tokenizer is provided by mistral-common and will be loaded from the base model.")

print(f"\nLoRA saved to: {FINAL_DIR}")
