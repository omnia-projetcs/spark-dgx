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
