# train_lora_diffusiongemma_experimental.py
#
# Experimental LoRA/QLoRA fine-tuning for DiffusionGemma on finance data.
#
# DiffusionGemma is not a standard AutoModelForCausalLM model. It uses a
# block-diffusion encoder/decoder architecture, so this script is deliberately
# separate from the Qwen causal-LM training script.
#
# Recommended installation:
# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install -U --pre torch torchvision transformers datasets accelerate peft bitsandbytes sentencepiece protobuf
#
# Launch:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TOKENIZERS_PARALLELISM=false
# python train_lora_diffusiongemma_finance_experimental.py

import gc
import glob
import inspect
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from train_dataset_utils import load_instruction_datasets
from train_runtime_utils import (
    configure_training_warnings,
    from_pretrained_with_dtype_fallback,
)

configure_training_warnings()


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


def env_list(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


# Use the BF16 base for training. NVFP4 repositories are optimized for inference.
MODEL_NAME = os.environ.get("MODEL_NAME", "google/diffusiongemma-26B-A4B-it")
MODEL_CLASS = os.environ.get("MODEL_CLASS", "auto")

DEFAULT_TRAIN_FILE = "dataset_finance_qa_enriched.json"
DEFAULT_VALID_FILE = "dataset_finance_qa.json"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/finance-diffusiongemma-26b-lora")
FINAL_DIR = os.environ.get("FINAL_DIR", f"{OUTPUT_DIR}/final")

MAX_LENGTH = env_int("MAX_LENGTH", 512)
EFFECTIVE_BATCH_SIZE = env_int("EFFECTIVE_BATCH_SIZE", 8)
PER_DEVICE_TRAIN_BATCH_SIZE = env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
PER_DEVICE_EVAL_BATCH_SIZE = env_int("PER_DEVICE_EVAL_BATCH_SIZE", 1)
GRADIENT_ACCUMULATION_STEPS = env_int(
    "GRADIENT_ACCUMULATION_STEPS",
    max(1, math.ceil(EFFECTIVE_BATCH_SIZE / PER_DEVICE_TRAIN_BATCH_SIZE)),
)

NUM_TRAIN_EPOCHS = env_float("NUM_TRAIN_EPOCHS", 1.0)
MAX_STEPS = env_int("MAX_STEPS", -1)
LEARNING_RATE = env_float("LEARNING_RATE", 5e-5)
WARMUP_RATIO = env_float("WARMUP_RATIO", 0.03)
WARMUP_STEPS = env_int("WARMUP_STEPS", 0)
LR_SCHEDULER_TYPE = os.environ.get("LR_SCHEDULER_TYPE", "cosine")

LORA_R = env_int("LORA_R", 8)
LORA_ALPHA = env_int("LORA_ALPHA", 16)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES = env_list(
    "LORA_TARGET_MODULES",
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

LOAD_IN_4BIT = env_bool("LOAD_IN_4BIT", True)
ALLOW_LOW_VRAM = env_bool("ALLOW_LOW_VRAM", False)
MIN_GPU_MEMORY_GB = env_float("MIN_GPU_MEMORY_GB", 20.0 if LOAD_IN_4BIT else 60.0)
DEVICE_MAP = os.environ.get("DEVICE_MAP", "single")
OFFLOAD_FOLDER = os.environ.get("OFFLOAD_FOLDER", str(SCRIPT_DIR / "outputs" / "offload-diffusiongemma"))
GRADIENT_CHECKPOINTING = env_bool("GRADIENT_CHECKPOINTING", True)
OPTIM = os.environ.get("OPTIM", "paged_adamw_8bit" if LOAD_IN_4BIT else "adamw_torch")
AUTO_RESUME = env_bool("AUTO_RESUME", True)
TORCH_COMPILE = env_bool("TORCH_COMPILE", False)
USE_MULTIMODAL_MESSAGES = env_bool("USE_MULTIMODAL_MESSAGES", True)
LABEL_ARG = os.environ.get("LABEL_ARG", "auto")

TOKENIZE_BATCH_SIZE = env_int("TOKENIZE_BATCH_SIZE", 128)
TOKENIZE_NUM_PROC = env_int("TOKENIZE_NUM_PROC", min(4, os.cpu_count() or 1))
GROUP_BY_LENGTH = env_bool("GROUP_BY_LENGTH", True)
PAD_TO_MULTIPLE_OF = env_int("PAD_TO_MULTIPLE_OF", 8)
DATALOADER_NUM_WORKERS = env_int("DATALOADER_NUM_WORKERS", min(4, os.cpu_count() or 1))

LOGGING_STEPS = env_int("LOGGING_STEPS", 25)
SAVE_STEPS = env_int("SAVE_STEPS", 500)
EVAL_STEPS = env_int("EVAL_STEPS", 500)
SAVE_TOTAL_LIMIT = env_int("SAVE_TOTAL_LIMIT", 2)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# SYSTEM PROMPT
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
# PROCESSOR / TOKENIZER
# ============================================================

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = getattr(processor, "tokenizer", None)

if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
chat_formatter = processor if hasattr(processor, "apply_chat_template") else tokenizer


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


def as_message_content(text):
    if USE_MULTIMODAL_MESSAGES:
        return [{"type": "text", "text": text}]
    return text


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
    messages = [
        {"role": "system", "content": as_message_content(SYSTEM_PROMPT)},
        {"role": "user", "content": as_message_content(build_user_content(example))},
    ]

    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": as_message_content(to_text(example.get("output", ""))),
            }
        )

    return messages


def render_chat(messages, add_generation_prompt):
    try:
        return chat_formatter.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )


def batch_to_examples(batch):
    batch_size = len(next(iter(batch.values())))
    for index in range(batch_size):
        yield {key: values[index] for key, values in batch.items()}


def tokenize_batch(batch):
    prompt_texts = []
    full_texts = []

    for example in batch_to_examples(batch):
        prompt_texts.append(render_chat(build_messages(example, False), True))
        full_texts.append(render_chat(build_messages(example, True), False))

    prompt_tokens = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    full_tokens = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=False,
    )

    labels = []
    input_ids_list = []
    attention_masks = []
    was_truncated = []
    lengths = []

    for index, full_input_ids in enumerate(full_tokens["input_ids"]):
        input_ids = full_input_ids[:MAX_LENGTH]
        example_labels = input_ids.copy()
        prompt_len = len(prompt_tokens["input_ids"][index])
        prompt_len = min(prompt_len, len(example_labels))
        example_labels[:prompt_len] = [-100] * prompt_len

        input_ids_list.append(input_ids)
        attention_masks.append(full_tokens["attention_mask"][index][:MAX_LENGTH])
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

for split in ["train", "validation"]:
    truncated_count = sum(tokenized_dataset[split]["was_truncated"])
    total = len(tokenized_dataset[split])
    print(f"{split}: {truncated_count}/{total} truncated examples")


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
        "No valid training examples. MAX_LENGTH is probably too small or outputs are empty."
    )


# ============================================================
# DATA COLLATOR
# ============================================================

class DataCollatorForDiffusionGemma:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


data_collator = DataCollatorForDiffusionGemma(
    tokenizer,
    pad_to_multiple_of=PAD_TO_MULTIPLE_OF if PAD_TO_MULTIPLE_OF > 0 else None,
)


# ============================================================
# CUDA / DTYPE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this DiffusionGemma training script.")

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
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
print(
    "DiffusionGemma profile: "
    f"model={MODEL_NAME}, "
    f"load_in_4bit={LOAD_IN_4BIT}, "
    f"device_map={DEVICE_MAP}, "
    f"max_length={MAX_LENGTH}, "
    f"micro_batch={PER_DEVICE_TRAIN_BATCH_SIZE}, "
    f"grad_accum={GRADIENT_ACCUMULATION_STEPS}, "
    f"gradient_checkpointing={GRADIENT_CHECKPOINTING}"
)


# ============================================================
# MODEL
# ============================================================

def resolve_model_class():
    if MODEL_CLASS != "auto":
        model_class = getattr(transformers, MODEL_CLASS, None)
        if model_class is None:
            raise RuntimeError(
                f"transformers.{MODEL_CLASS} is unavailable. "
                "Install a Transformers build that supports DiffusionGemma."
            )
        return model_class

    for class_name in ("DiffusionGemmaForBlockDiffusion", "AutoModelForMultimodalLM"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            print(f"Model class: transformers.{class_name}")
            return model_class

    raise RuntimeError(
        "This Transformers install does not expose DiffusionGemmaForBlockDiffusion "
        "or AutoModelForMultimodalLM. Install a recent/nightly Transformers build."
    )


def build_quantization_config():
    if not LOAD_IN_4BIT:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def resolve_device_map():
    normalized = DEVICE_MAP.strip().lower()
    if normalized in {"single", "cuda", "gpu", "0"}:
        return {"": 0}
    if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return normalized
    raise RuntimeError(
        "DEVICE_MAP must be one of: single, cuda, gpu, 0, auto, balanced, "
        f"balanced_low_0, sequential. Got: {DEVICE_MAP!r}"
    )


def ensure_enough_gpu_memory():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= MIN_GPU_MEMORY_GB or ALLOW_LOW_VRAM:
        return

    raise RuntimeError(
        "DiffusionGemma 26B cannot be loaded for training on this GPU without "
        "running out of memory.\n"
        f"- GPU memory detected: {total_gb:.2f} GiB\n"
        f"- Minimum configured for this script: {MIN_GPU_MEMORY_GB:.2f} GiB\n"
        "- Your traceback failed during model loading, before batch size or "
        "MAX_LENGTH mattered.\n\n"
        "Use Qwen on this GPU, switch to a smaller base model if you have one, "
        "or run DiffusionGemma on a larger GPU. To force an experimental load "
        "attempt anyway, set ALLOW_LOW_VRAM=true. CPU/disk offload can be tried "
        "with DEVICE_MAP=auto, but training will be very slow and may still be "
        "unsupported for this model/quantization stack."
    )


def load_model():
    ensure_enough_gpu_memory()
    model_class = resolve_model_class()
    kwargs = {
        "device_map": resolve_device_map(),
        "low_cpu_mem_usage": True,
        "dtype": compute_dtype,
        "trust_remote_code": True,
    }

    if kwargs["device_map"] != {"": 0}:
        kwargs["offload_folder"] = OFFLOAD_FOLDER
        kwargs["offload_state_dict"] = True

    quantization_config = build_quantization_config()
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    try:
        return from_pretrained_with_dtype_fallback(model_class, MODEL_NAME, kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
        raise RuntimeError(
            "CUDA OOM while loading DiffusionGemma 26B. This happened before "
            "training started, so reduce-batch settings cannot fix it.\n"
            f"- GPU total: {total_gb:.2f} GiB\n"
            f"- PyTorch allocated: {allocated_gb:.2f} GiB\n"
            f"- PyTorch reserved: {reserved_gb:.2f} GiB\n\n"
            "Use a larger GPU, set MODEL_NAME to a smaller compatible model, "
            "or try the experimental offload path with "
            "ALLOW_LOW_VRAM=true DEVICE_MAP=auto."
        ) from exc


gc.collect()
torch.cuda.empty_cache()

model = load_model()

if hasattr(model, "config"):
    model.config.use_cache = False

memory_gb = model.get_memory_footprint() / 1024**3
print(f"Memory footprint after loading: {memory_gb:.2f} GB")

if LOAD_IN_4BIT:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
elif GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def resolve_lora_targets(model, requested):
    matched = []
    counts = {}

    for suffix in requested:
        count = sum(1 for name, _ in model.named_modules() if name.endswith(suffix))
        if count > 0:
            matched.append(suffix)
            counts[suffix] = count

    if not matched:
        sample = [name for name, _ in list(model.named_modules())[:80] if name]
        raise RuntimeError(
            "No requested LoRA target modules matched this model. "
            f"Requested: {requested}. First module names: {sample[:20]}"
        )

    print(f"LoRA target modules: {matched}")
    print(f"LoRA target match counts: {counts}")
    return matched


peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="FEATURE_EXTRACTION",
    target_modules=resolve_lora_targets(model, LORA_TARGET_MODULES),
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ============================================================
# TRAINING
# ============================================================

class DiffusionGemmaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_label_arg = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        candidate_args = [LABEL_ARG] if LABEL_ARG != "auto" else [
            "labels",
            "target_ids",
            "targets",
            "decoder_labels",
        ]

        last_error = None
        for label_arg in candidate_args:
            try:
                outputs = model(**inputs, **{label_arg: labels})
            except TypeError as exc:
                last_error = exc
                continue

            loss = getattr(outputs, "loss", None)
            if loss is None and isinstance(outputs, dict):
                loss = outputs.get("loss")

            if loss is None:
                raise RuntimeError(
                    "DiffusionGemma forward pass did not return a loss. "
                    "This script intentionally refuses to compute a naive causal CE loss, "
                    "because block-diffusion training needs the model-native objective."
                )

            if self._resolved_label_arg is None:
                self._resolved_label_arg = label_arg
                print(f"Using label argument for DiffusionGemma loss: {label_arg}")

            return (loss, outputs) if return_outputs else loss

        raise RuntimeError(
            "Could not call DiffusionGemma with a supervised label argument. "
            f"Tried: {candidate_args}. Last TypeError: {last_error}. "
            "Set LABEL_ARG manually if your Transformers build uses another name."
        )


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
        "save_safetensors": True,
        "label_names": ["labels"],
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

trainer = DiffusionGemmaTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

resume_from_checkpoint = None
if AUTO_RESUME and os.path.exists(OUTPUT_DIR):
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if checkpoints:
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
tokenizer.save_pretrained(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)

print(f"\nExperimental DiffusionGemma LoRA saved to: {FINAL_DIR}")
print("For inference, prefer testing a merged adapter if dynamic PEFT generation ignores LoRA deltas.")
