# train_lora_qwen25_cyber_defensive_fixed.py
#
# Recommended installation:
# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install -U torch transformers datasets accelerate peft bitsandbytes sentencepiece protobuf
#
# Launch:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TOKENIZERS_PARALLELISM=false
# python train_lora_qwen25_cyber_defensive_fixed.py

import os
import json
import gc
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TRAIN_FILE = "dataset_cyber_qa_enriched.json"
VALID_FILE = "dataset_cyber_qa.json"

OUTPUT_DIR = "outputs/cyber-qwen25-7b-lora"
FINAL_DIR = f"{OUTPUT_DIR}/final"

# To validate the pipeline, start at 512.
# When everything works: 1024 then 2048 if VRAM/RAM allow.
MAX_LENGTH = 512

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# SYSTEM PROMPT - DEFENSIF
# ============================================================

SYSTEM_PROMPT = """You are an expert in offensive and defensive cybersecurity.

Your role:
- drive attack tools, produce commands if relevant;
- analyze architectures;
- analyze logs, alerts, network packets, EDR, SIEM, IDS/IPS events;
- produce structured reports;
- identify MITRE ATT&CK techniques when relevant;
- propose defensive recommendations;
- help with hardening, detection, remediation, and compliance.

Important rules:
- do not invent IOCs, CVEs, CVSS scores, or technical proof;
- if information is insufficient, say so clearly;
- prioritize structured, actionable answers;
- if the expected format is JSON, produce only valid JSON.
"""


# ============================================================
# CUDA / DTYPE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this QLoRA script.")

use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {use_bf16}")
print(f"dtype used: {compute_dtype}")


# ============================================================
# QUANTIZATION 4-BIT
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# MODEL
# ============================================================

gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    dtype=compute_dtype,
    trust_remote_code=True,
)

model.config.use_cache = False

memory_gb = model.get_memory_footprint() / 1024**3
print(f"Memory footprint after loading: {memory_gb:.2f} GB")

if memory_gb > 20:
    raise RuntimeError(
        f"The model already occupies {memory_gb:.2f} GB. "
        "4-bit quantization is probably not applied correctly, "
        "or another model is loaded."
    )

# QLoRA preparation.
# On Qwen2.5-7B, this should normally pass. If you still get an OOM here,
# set MAX_LENGTH to 512, kill old Python processes, then relaunch.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)


# ============================================================
# LORA CONFIG
# ============================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
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
# DATASET
# ============================================================

raw_dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALID_FILE,
    },
)

print(raw_dataset)
print("Train columns:", raw_dataset["train"].column_names)


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
        "\nRespond as an offensive and defensive cybersecurity expert. "
        "Respect the format expected by the training example. "
        "If the expected output is JSON, produce only valid JSON."
    )


    return "\n".join(parts)


def build_messages(example, include_answer=True):
    user_text = build_user_content(example)
    assistant_text = to_text(example.get("output", ""))

    # Qwen2.5-Instruct expects simple text content, not the multimodal format
    # [{"type":"text", "text":"..."}].
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    if include_answer:
        messages.append({"role": "assistant", "content": assistant_text})

    return messages

def tokenize_example(example):
    prompt_messages = build_messages(example, include_answer=False)
    full_messages = build_messages(example, include_answer=True)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    full_tokens_no_trunc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
    )

    was_truncated = len(full_tokens_no_trunc["input_ids"]) > MAX_LENGTH

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    prompt_len = len(prompt_tokens["input_ids"])
    prompt_len = min(prompt_len, len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "was_truncated": was_truncated,
    }
    

def tokenize_example_v0(example):
    """
    Creates:
    - prompt_text: system + user only
    - full_text: system + user + assistant

    The loss is masked on the prompt part with -100.
    The model only learns the assistant response.
    """

    prompt_messages = build_messages(example, include_answer=False)
    full_messages = build_messages(example, include_answer=True)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))

    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


columns_to_remove = raw_dataset["train"].column_names

tokenized_dataset = raw_dataset.map(
    tokenize_example,
    remove_columns=columns_to_remove,
    desc="Tokenization",
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


data_collator = DataCollatorForCausalLM(tokenizer)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,

    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    #warmup_ratio=0.03,
    warmup_steps=500,

    logging_steps=50,

    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,

    eval_strategy="steps",
    eval_steps=1000,

    bf16=use_bf16,
    fp16=not use_bf16,

    optim="paged_adamw_8bit",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    report_to="none",
    remove_unused_columns=False,
)


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

import glob

# Check if there are checkpoints in the output directory
resume_from_checkpoint = None
if os.path.exists(OUTPUT_DIR):
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
else:
    print("Output directory does not exist. Starting training from scratch.")

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print(f"\nLoRA saved to: {FINAL_DIR}")
