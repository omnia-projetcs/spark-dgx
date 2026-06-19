# Fine-Tuning Qwen 2.5 for Cybersecurity (QLoRA)

This directory contains the pipeline to fine-tune **Qwen/Qwen2.5-7B-Instruct** using QLoRA (Quantized Low-Rank Adaptation) on cybersecurity QA datasets.

The training script leverages 4-bit NormalFloat (NF4) quantization, 8-bit Paged AdamW optimization, and gradient checkpointing to allow fine-tuning on consumer-grade and enterprise GPUs (with low VRAM footprints, e.g. < 24GB).

---

## Prerequisites & Installation

It is recommended to run the training within a dedicated virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -U torch transformers datasets accelerate peft bitsandbytes sentencepiece protobuf
```

Ensure you have a CUDA-compatible GPU setup. The script automatically detects and uses `bfloat16` if your hardware supports it, or falls back to `float16`.

---

## Dataset Structure

The script loads the training and validation data locally in JSON format from the workspace:
- **Training dataset**: `dataset_cyber_qa_enriched.json`
- **Validation dataset**: `dataset_cyber_qa.json`

The JSON files should follow the standard instruction-following format:
```json
[
  {
    "instruction": "Explain how to mitigate SQL injection.",
    "input": "",
    "output": "To mitigate SQL injection, use parameterized queries (prepared statements)..."
  }
]
```

### Prompt Template & Prompt Masking
The dataset is formatted using a specialized cybersecurity system prompt (defining the assistant as an offensive and defensive cybersecurity expert). 
To ensure the model learns only the assistant's responses and not the prompts:
- The input tokens (system prompt + user instruction) are masked with a label of `-100`.
- The loss is computed only on the assistant's output tokens.

---

## Training Configurations

The key hyperparameters configured in `train_lora_qwen25_cyber_defensive_fixed_v2.py` are:

- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Max Sequence Length**: `512` (can be increased to `1024` or `2048` if GPU memory allows)
- **Quantization**: 4-bit double quantization (NF4)
- **LoRA Hyperparameters**:
  - Rank ($r$): `8`
  - Alpha ($\alpha$): `16`
  - Dropout: `0.05`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all linear modules of the attention and MLP layers)
- **Optimizer**: `paged_adamw_8bit`
- **Learning Rate**: `1e-4` with a cosine schedule and `500` warmup steps
- **Batch Size**: Effective batch size of `16` (Per-device batch size `1` * Gradient accumulation steps `16`)
- **Epochs**: `1`

---

## Running the Training

Prior to launching the script, set the environment variables to optimize memory allocation and disable tokenizer warnings:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

### VRAM and Resource Constraints
- The script includes safety checks. If the base model takes more than 20 GB of VRAM right after loading, training will halt (indicating that 4-bit quantization did not load correctly).
- If you encounter Out-Of-Memory (OOM) errors during the backward pass:
  1. Kill any stale python processes (`pkill -f python`).
  2. Reduce `MAX_LENGTH` to `512` or lower.
  3. Ensure `expandable_segments:True` is set in `PYTORCH_CUDA_ALLOC_CONF`.

---

## Outputs & Artifacts

Once training completes, the LoRA adapters and tokenizer configurations will be saved to:
- **Adapter Directory**: `outputs/cyber-qwen25-7b-lora/final`

You can load these adapters on top of the base model using the Hugging Face `peft` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "outputs/cyber-qwen25-7b-lora/final"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
```
