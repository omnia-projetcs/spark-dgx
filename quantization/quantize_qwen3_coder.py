#pip install llmcompressor datasets --break-system-packages

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

model_id: str = "Qwen/Qwen3-Coder-Next"
output_dir_path: str = "Qwen3-Coder-Next-AWQ-4bit"

# Load tokenizer to format chat/code samples with correct template
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess(example):
    if "input" in example and "output" in example:
        messages = list(example["input"])
        messages.append({"role": "assistant", "content": example["output"]})
    elif "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        messages = []
        for msg in example["conversations"]:
            role = "user" if msg["from"] in ["human", "user"] else "assistant"
            messages.append({"role": role, "content": msg["value"]})
    elif "instruction" in example:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example.get("output", example.get("response", ""))}
        ]
    elif "text" in example:
        return {"text": example["text"]}
    else:
        raise ValueError(f"Unknown sample keys: {example.keys()}")
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception:
        # Fallback to simple concatenation if tokenizer has no chat template
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"
            
    return {"text": text}

# Load a slice of the 'code' split for coding model calibration
calibration_set = load_dataset(
    "nvidia/Llama-Nemotron-Post-Training-Dataset", split="code[:2048]"
)

# Apply preprocessing
calibration_set = calibration_set.map(preprocess)

# Tokenize inputs
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=2048,
        truncation=True,
        add_special_tokens=False,
    )

calibration_set = calibration_set.map(tokenize, remove_columns=calibration_set.column_names)

recipe = [AWQModifier(
    ignore=["lm_head"],
    config_groups={"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 4, "type": "int",
                    "symmetric": False, "strategy": "group", "group_size": 32}
    }}
)]

oneshot(
    model=model_id,
    dataset=calibration_set,
    recipe=recipe,
    output_dir=output_dir_path,
    max_seq_length=2048,
    num_calibration_samples=1024,
)

