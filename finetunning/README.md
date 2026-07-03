# Fine-Tuning LoRA / QLoRA

This directory contains LoRA/QLoRA training scripts for Qwen 2.5, Mistral-Small-3.2, and DiffusionGemma on local instruction datasets.

The training scripts leverage 4-bit NormalFloat (NF4) quantization, 8-bit Paged AdamW optimization, dynamic padding, length-grouped batches, batched tokenization, TF32 when available, and automatic FlashAttention 2 -> SDPA fallback. The Qwen defaults favor 16 GB GPUs: micro-batch `1` with gradient checkpointing enabled. Larger GPUs can opt into the fast profile below.

---

## Prerequisites & Installation

It is recommended to run the training within a dedicated virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -U torch torchvision transformers datasets accelerate peft bitsandbytes sentencepiece protobuf pillow mistral-common
```

Ensure you have a CUDA-compatible GPU setup. The script automatically detects and uses `bfloat16` if your hardware supports it, or falls back to `float16`.

Optional: if your platform supports it, install FlashAttention 2 for faster attention kernels. The script will try it automatically and fall back to PyTorch SDPA if it is unavailable.

---

## Dataset Structure

By default, the cybersecurity scripts load the cybersecurity JSON files from this directory:
- **Training dataset**: `dataset_cyber_qa_enriched.json`
- **Validation dataset**: `dataset_cyber_qa.json`

The finance scripts load the finance JSON files from this directory:
- **Training dataset**: `dataset_finance_qa_enriched.json`
- **Validation dataset**: `dataset_finance_qa.json`

For faster dataset iteration, the training scripts also support multi-file loading without editing Python code:

```bash
# Train on every enriched local QA dataset and create validation automatically.
DATASET_GLOB="dataset_*_qa_enriched.json" \
VALIDATION_SPLIT=0.05 \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

```bash
# Explicit train/validation files. Globs are accepted too.
TRAIN_FILES="dataset_cyber_qa_enriched.json,dataset_finance_qa_enriched.json" \
VALID_FILES="dataset_cyber_qa.json,dataset_finance_qa.json" \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

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

The loader also accepts common aliases to make imported datasets quicker to add:
- prompt fields: `instruction`, `prompt`, `question`, `query`, `task`
- optional context fields: `input`, `context`, `ctx`, `data`
- answer fields: `output`, `answer`, `response`, `completion`
- simple chat fields: `messages` or `conversations` with user/human and assistant/gpt turns

Useful dataset variables:

| Variable | Default | Use |
|---|---:|---|
| `TRAIN_FILE` | `dataset_cyber_qa_enriched.json` | Single training file, kept for backward compatibility. |
| `VALID_FILE` | `dataset_cyber_qa.json` | Single validation file, kept for backward compatibility. |
| `TRAIN_FILES` | unset | Comma-separated training files or globs. |
| `VALID_FILES` | unset | Comma-separated validation files or globs. |
| `DATASET_FILES` | unset | Comma-separated train files; validation is auto-split unless `VALID_FILES` is set. |
| `DATASET_GLOB` | unset | One or more comma-separated train globs; validation is auto-split unless `VALID_FILES` is set. |
| `DATASET_DIR` | script directory | Base directory for relative dataset paths. |
| `VALIDATION_SPLIT` | `0.05` | Fraction used for automatic validation split. |
| `DATASET_SEED` | `42` | Shuffle/split seed. |
| `MAX_TRAIN_SAMPLES` | `0` | Limit training examples for quick smoke tests. `0` means no limit. |
| `MAX_VALID_SAMPLES` | `0` | Limit validation examples for quick smoke tests. `0` means no limit. |

### Prompt Template & Prompt Masking
The cyber scripts use a specialized cybersecurity system prompt. The finance scripts use a finance and market analysis system prompt.
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
- **Learning Rate**: `1e-4` with a cosine schedule and `0.03` warmup ratio
- **Batch Size**: Effective batch size of `16` by default (per-device batch size `1` * gradient accumulation steps `16`)
- **Epochs**: `1`
- **Speed Optimizations**:
  - tokenization runs before model loading, in batches, with up to `4` CPU processes;
  - dynamic padding pads to multiples of `8` for Tensor Core-friendly batches;
  - `group_by_length=True` reduces wasted padding;
  - `TF32` is enabled on Ampere/Blackwell-class GPUs;
  - attention uses `flash_attention_2` if available, otherwise `sdpa`;
  - gradient checkpointing is enabled by default to keep Qwen 2.5 training inside 16 GB VRAM.

---

## Running the Training

Prior to launching the script, set the environment variables to optimize memory allocation and disable tokenizer warnings:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

### Fast Profiles

Default 16 GB-safe profile:
```bash
cd finetunning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

Quick throughput benchmark before a full run:
```bash
MAX_STEPS=200 \
SAVE_STEPS=1000000 \
EVAL_STEPS=1000000 \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

Fast dataset smoke test after adding new JSON files:
```bash
DATASET_GLOB="dataset_*_qa_enriched.json" \
VALIDATION_SPLIT=0.05 \
MAX_TRAIN_SAMPLES=5000 \
MAX_VALID_SAMPLES=500 \
MAX_STEPS=100 \
SAVE_STEPS=1000000 \
EVAL_STEPS=1000000 \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

Fast profile if VRAM allows it:
```bash
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
EFFECTIVE_BATCH_SIZE=16 \
GRADIENT_CHECKPOINTING=false \
MAX_LENGTH=512 \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

Extra conservative profile if you still hit OOM:
```bash
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
GRADIENT_CHECKPOINTING=true \
ATTN_IMPLEMENTATION=sdpa \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

Longer context profile:
```bash
MAX_LENGTH=1024 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
GRADIENT_CHECKPOINTING=true \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

### Useful Runtime Knobs

| Variable | Default | Use |
|---|---:|---|
| `MAX_LENGTH` | `512` | Sequence length. Increase to `1024`/`2048` only if memory allows. |
| `MAX_STEPS` | `-1` | Stop early for a benchmark. Use `200` before a full 160k-record run. |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | Micro-batch size. Keep `1` on 16 GB GPUs; raise only when VRAM allows. |
| `EFFECTIVE_BATCH_SIZE` | `16` | Target effective batch; auto-computes gradient accumulation unless overridden. |
| `GRADIENT_ACCUMULATION_STEPS` | auto | Set directly for exact control. |
| `GRADIENT_CHECKPOINTING` | `true` | Saves VRAM but slows training; set `false` only for larger GPUs. |
| `ATTN_IMPLEMENTATION` | `auto` | `auto` tries `flash_attention_2`, then `sdpa`; set `sdpa` to skip the FlashAttention attempt. |
| `TOKENIZE_NUM_PROC` | up to `4` | CPU workers for preprocessing. |
| `TOKENIZE_BATCH_SIZE` | `256` | Batch size for tokenizer calls. Lower if RAM is tight. |
| `DATALOADER_NUM_WORKERS` | up to `4` | PyTorch dataloader workers. |
| `OPTIM` | `paged_adamw_8bit` | Keep for QLoRA memory efficiency; try `adamw_torch_fused` only if your install supports it and VRAM is comfortable. |
| `TF32` | auto | Enable/disable TensorFloat-32 matmul on compatible NVIDIA GPUs. |
| `AUTO_RESUME` | `true` | Resume latest checkpoint automatically. |
| `TRAIN_FILES` / `VALID_FILES` | unset | Add several local JSON files without changing code. |
| `DATASET_GLOB` | unset | Quickly train on files matching a pattern. |
| `MAX_TRAIN_SAMPLES` / `MAX_VALID_SAMPLES` | `0` | Cap examples for smoke tests. |

### VRAM and Resource Constraints
- The script includes safety checks. If the base model takes more than 20 GB of VRAM right after loading, training will halt (indicating that 4-bit quantization did not load correctly).
- If you encounter Out-Of-Memory (OOM) errors during the forward, loss, or backward pass:
  1. Kill any stale python processes (`pkill -f python`).
  2. Keep `PER_DEVICE_TRAIN_BATCH_SIZE=1`.
  3. Keep `GRADIENT_CHECKPOINTING=true`.
  4. Reduce `MAX_LENGTH` to `384` or `256`.
  5. Ensure `expandable_segments:True` is set in `PYTORCH_CUDA_ALLOC_CONF`.

---

## Finance Dataset Training Scripts

Three dedicated finance training entrypoints are available. Run the one you want to train:

```bash
cd finetunning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Qwen 2.5
python train_lora_qwen25_finance.py

# Mistral-Small-3.2
python train_lora_mistral_small32_finance.py

# DiffusionGemma
python train_lora_diffusiongemma_finance_experimental.py
```

Default finance outputs:
- **Qwen 2.5**: `outputs/finance-qwen25-7b-lora/final`
- **Mistral-Small-3.2**: `outputs/finance-mistral-small3.2-lora/final`
- **DiffusionGemma**: `outputs/finance-diffusiongemma-26b-lora/final`

The Mistral-Small-3.2 script uses `Mistral3ForConditionalGeneration` and `mistral-common` tokenization, matching the Hugging Face model card.

Quick smoke test example:

```bash
MAX_STEPS=10 \
MAX_TRAIN_SAMPLES=20 \
MAX_VALID_SAMPLES=10 \
python train_lora_mistral_small32_finance.py
```

---

## Mistral-Small-3.2 LoRA / QLoRA

Mistral-Small-3.2 is a 24B parameter model. Due to its size, fine-tuning requires 4-bit QLoRA and aggressive gradient checkpointing to run on standard GPUs.

Use the dedicated script:

```bash
cd finetunning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python train_lora_mistral_small32_cyber_defensive.py
```

### Config & Memory Profiles

- **Base model**: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- **Model class**: `transformers.Mistral3ForConditionalGeneration`
- **Tokenizer**: `mistral-common` `MistralTokenizer`
- **Output directory**: `outputs/cyber-mistral-small3.2-lora`
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Gradient checkpointing**: enabled by default (`GRADIENT_CHECKPOINTING=true`) to save memory.
- **Batch size**: default per-device batch size `1`, effective batch size `16` (via auto-computed gradient accumulation steps `16`).

If you have a large GPU (e.g. 80 GB A100/H100), you can increase the per-device batch size:
```bash
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
python train_lora_mistral_small32_cyber_defensive.py
```

If memory is extremely tight:
```bash
MAX_LENGTH=256 \
python train_lora_mistral_small32_cyber_defensive.py
```

To run a quick test:
```bash
MAX_STEPS=10 \
MAX_TRAIN_SAMPLES=20 \
MAX_VALID_SAMPLES=10 \
python train_lora_mistral_small32_cyber_defensive.py
```

---

## Experimental: DiffusionGemma LoRA / QLoRA

DiffusionGemma is not a standard causal language model. It uses a block-diffusion encoder/decoder architecture, so the Qwen script must not be reused by only changing `MODEL_NAME`.

Use the dedicated experimental script:

```bash
cd finetunning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python train_lora_diffusiongemma_experimental.py
```

Default profile:
- **Base model**: `google/diffusiongemma-26B-A4B-it`
- **Training mode**: experimental PEFT LoRA with `task_type="FEATURE_EXTRACTION"`
- **Quantization**: `LOAD_IN_4BIT=true` by default for QLoRA-style memory savings
- **Batch**: per-device batch size `1`, effective batch `8`
- **Gradient checkpointing**: enabled by default
- **VRAM guard**: the script refuses to load by default below `MIN_GPU_MEMORY_GB=20` GiB because 16 GB cards OOM while loading the 26B model, before training starts.

Useful profiles:

```bash
# Quick smoke test before a full run
MAX_STEPS=50 \
SAVE_STEPS=1000000 \
EVAL_STEPS=1000000 \
python train_lora_diffusiongemma_experimental.py
```

```bash
# Safer BF16 LoRA if memory allows it
LOAD_IN_4BIT=false \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 \
python train_lora_diffusiongemma_experimental.py
```

```bash
# Lowest training-memory profile. This does not reduce model-load VRAM.
LOAD_IN_4BIT=true \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
MAX_LENGTH=512 \
python train_lora_diffusiongemma_experimental.py
```

```bash
# Experimental CPU/disk offload attempt. Very slow and may still be unsupported.
ALLOW_LOW_VRAM=true \
DEVICE_MAP=auto \
python train_lora_diffusiongemma_experimental.py
```

```bash
# Use the abliterated BF16 variant as a starting point
MODEL_NAME=edwixx/diffusiongemma-26B-A4B-it-HERETIC-Uncensored \
LOAD_IN_4BIT=true \
python train_lora_diffusiongemma_experimental.py
```

Important notes:
- `nvidia/diffusiongemma-26B-A4B-IT-NVFP4` is best treated as an inference-optimized vLLM/NVFP4 artifact. For fine-tuning, prefer the BF16 base or a BF16 fine-tune.
- On a 16 GB GPU, use the Qwen 2.5 script or a smaller compatible base model. Lowering `MAX_LENGTH` or batch size cannot fix an OOM that happens during `from_pretrained`.
- You may need a very recent or pre-release `transformers` build. The script looks for `DiffusionGemmaForBlockDiffusion` first, then `AutoModelForMultimodalLM`.
- The script intentionally refuses to invent a naive causal loss if the model does not return a native training loss. Block-diffusion training should use the model-native objective.
- Dynamic PEFT adapters may need extra validation for generation because DiffusionGemma has shared encoder/decoder weights. If inference ignores the LoRA delta, test a merged adapter.

The DiffusionGemma adapter is saved to:
- **Adapter Directory**: `outputs/cyber-diffusiongemma-26b-lora/final`

---

## Outputs & Artifacts

Once training completes, the LoRA adapters and tokenizer configurations will be saved to:
- **Adapter Directory (Qwen 2.5)**: `outputs/cyber-qwen25-7b-lora/final`
- **Adapter Directory (Mistral-Small-3.2)**: `outputs/cyber-mistral-small3.2-lora/final`
- **Adapter Directory (DiffusionGemma)**: `outputs/cyber-diffusiongemma-26b-lora/final`
- **Adapter Directory (Finance Qwen 2.5)**: `outputs/finance-qwen25-7b-lora/final`
- **Adapter Directory (Finance Mistral-Small-3.2)**: `outputs/finance-mistral-small3.2-lora/final`
- **Adapter Directory (Finance DiffusionGemma)**: `outputs/finance-diffusiongemma-26b-lora/final`

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

### Export vers Hugging Face Hub

Vous pouvez pousser vos adaptateurs LoRA (ou le modèle fusionné) directement sur le [Hugging Face Hub](https://huggingface.co).

#### 1. Authentification avec votre token HF

Générez un token d'accès avec les droits **write** sur [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), puis exportez-le dans votre environnement :

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Ou connectez-vous de manière persistante via la CLI :

```bash
pip install -U huggingface_hub
huggingface-cli login --token "$HF_TOKEN"
```

#### 2. Pousser les adaptateurs LoRA (léger, ~100 Mo)

Ajoutez simplement `push_to_hub=True` à la sauvegarde de l'adaptateur, ou poussez-le après l'entraînement :

```python
from huggingface_hub import HfApi
import os

# Via l'API Python (après entraînement)
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path="outputs/cyber-qwen25-7b-lora/final",
    repo_id="votre-org/cyber-qwen25-7b-lora",   # remplacer par votre namespace
    repo_type="model",
)
```

Ou depuis le script de merge, directement avec `push_to_hub` :

```python
# À ajouter à la fin de merge_lora.py
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])

merged_model.push_to_hub("votre-org/cyber-qwen25-7b-merged")
tokenizer.push_to_hub("votre-org/cyber-qwen25-7b-merged")
```

#### 3. Variable d'environnement dans le script d'entraînement

Le script d'entraînement lit automatiquement `HF_TOKEN` si vous souhaitez activer `push_to_hub` dans `TrainingArguments` :

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HUB_MODEL_ID="votre-org/cyber-qwen25-7b-lora" \
python train_lora_qwen25_cyber_defensive_fixed_v2.py
```

> **Note** : ne committez jamais votre token en clair dans un fichier. Utilisez toujours une variable d'environnement ou un gestionnaire de secrets.

---

### Fusionner et Convertir en un Fichier Unique (Merge & Export to GGUF)

Si vous souhaitez fusionner les poids LoRA avec le modèle de base pour obtenir un seul modèle complet, puis le convertir au format de fichier unique **GGUF** (très utile pour l'exécution locale avec Ollama, LM Studio ou llama.cpp) :

#### 1. Fusionner le LoRA avec le modèle de base
Créez un script Python (par exemple `merge_lora.py`) :

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-7B-Instruct"
lora_weights_path = "outputs/cyber-qwen25-7b-lora/final"
output_dir = "outputs/qwen25-merged"

# 1. Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 2. Charger le modèle de base sur CPU (pour économiser la VRAM du GPU)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# 3. Charger le modèle LoRA
model = PeftModel.from_pretrained(base_model, lora_weights_path)

# 4. Fusionner les poids
merged_model = model.merge_and_unload()

# 5. Sauvegarder le modèle fusionné
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
```

#### 2. Convertir au format GGUF (Fichier Unique)
Pour convertir le dossier fusionné en un seul fichier `.gguf` :

1. Cloner `llama.cpp` et installer ses dépendances :
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   pip install -r requirements.txt
   ```

2. Convertir au format GGUF :
   ```bash
   # Convertir directement en format f16 (fichier unique lourd, ~14 Go)
   python convert_hf_to_gguf.py /chemin/vers/outputs/qwen25-merged --outfile mon_modele_f16.gguf
   
   # Ou quantifier le modèle pour réduire sa taille (ex: Q8_0 à ~7.7 Go, ou Q4_K_M à ~4.5 Go)
   make -j
   python convert_hf_to_gguf.py /chemin/vers/outputs/qwen25-merged --outfile temp.gguf
   ./llama-quantize temp.gguf mon_modele_Q8_0.gguf Q8_0
   rm temp.gguf
   ```

### Utilisation avec vLLM

vLLM prend en charge nativement le format Hugging Face standard (Safetensors) ainsi que le chargement dynamique des adaptateurs LoRA.

#### Option A : Lancer le modèle fusionné (Recommandé pour de meilleures performances)
Une fois le modèle fusionné (Étape 1), lancez simplement vLLM en pointant vers le dossier fusionné :
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /chemin/vers/outputs/qwen25-merged \
    --port 8000
```

#### Option B : Lancer avec chargement dynamique de LoRA (Sans fusionner les poids)
vLLM peut appliquer votre adaptateur LoRA à chaud sur le modèle de base. Lancez vLLM en spécifiant le modèle de base et le chemin de votre adaptateur :
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules cyber-defensive=/chemin/vers/outputs/cyber-qwen25-7b-lora/final \
    --port 8000
```
Lors de vos requêtes API, passez simplement `"model": "cyber-defensive"` dans le payload pour cibler votre modèle finetuné.
