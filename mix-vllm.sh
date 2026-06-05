#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# mix-vllm.sh — vLLM launcher for DGX Spark (GB10 / SM121, 128 GB)
# Models ranked by single-node throughput (Spark Arena leaderboard)
#
#  Rank  Model                                         tok/s    Quant     Image
#  ────  ──────────────────────────────────────────   ───────  ───────   ────────────
#  #1    AEON-7/Qwen3.6-35B-heretic-NVFP4 + DFlash    ~71       NVFP4     aeon-7 v1.2   ← best performance (~117 w/ DFlash)
#  #2    rdtand/Qwen3.6-35B-A3B-PrismaQuant            ~59      4.75bit   vllm-latest
#  #3    nvidia/Nemotron-3-Nano-30B-A3B-NVFP4          ~58      NVFP4     eugr-nightly
#  #4    bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4   ~50      NVFP4     eugr-nightly
#  ──    Neural-ICE/Gemma-4-E2B-it-NVFP4              ~120     NVFP4     eugr-nightly  ← multimodal, native audio, W4A4
#  ──    bg-digitalservices/Gemma-4-E2B-NVFP4         ~120     NVFP4     eugr-nightly  ← base model, W4A4
#  #5    Qwen/Qwen3.6-35B-A3B-FP8                      ~30      FP8       cu130-nightly
#  #6    rdtand/MiniMax-M2.7-PrismaQuant-3.20bit       ~25      3.20bit   eugr-nightly
#  #7    Intel/Qwen3-Coder-Next-int4-AutoRound         ~17      INT4      vllm-latest
#  #8    RedHatAI/Qwen3.5-122B-A10B-NVFP4              ~17      NVFP4     eugr-nightly  ← best quality
#  #9    nvidia/Nemotron-3-Super-120B-A12B-NVFP4       ~15      NVFP4     eugr-nightly
#  ──    LiquidAI/LFM2.5-350M                         ~212      BF16      vllm-latest   ← ultra-lightweight
#  ──    Qwen/Qwen3.5-0.8B                            ~103      BF16      vllm-latest   ← ultra-lightweight
#  ──    RedHatAI/Mistral-Small-24B-FP8               ~8.6      FP8       cu130-nightly ← low VRAM (~24 GB)
#  ──    dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4 fast   NVFP4     eugr-nightly
#
#  * gpt-oss MXFP4 uses eugr-nightly with CUTLASS backend (no local build needed)
# ─────────────────────────────────────────────────────────────────────────────

# ── HUGGING FACE TOKEN ────────────────────────────────────────────────────────
# Required to download gated / private models from the Hugging Face Hub.
#
# How to get your token:
#   1. Create an account on https://huggingface.co/join
#   2. Go to Settings → Access Tokens : https://huggingface.co/settings/tokens
#   3. Click "Create new token"
#   4. Choose a name (e.g. "dgx-spark") and select permission "Read"
#      (or "Fine-grained" with at least "Read access to contents of all
#       repos under your personal namespace")
#   5. Copy the token (starts with hf_...) and paste it below
#
# ⚠️  For gated models (Llama, Gemma, etc.) you must also accept the model
#     license on its Hugging Face page before downloading.
# ─────────────────────────────────────────────────────────────────────────────
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-hf_...}"

# ── TOKEN VALIDATION ──────────────────────────────────────────────────────────
# Check if the token is set. If not, print a prominent warning but do NOT exit.
# We will only exit later if a download actually fails due to lack of a token.
if [[ -z "${HUGGING_FACE_HUB_TOKEN}" || "${HUGGING_FACE_HUB_TOKEN}" == "hf_..." ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║  ⚠️  WARNING — Hugging Face token is NOT configured!                 ║"
  echo "╠══════════════════════════════════════════════════════════════════════╣"
  echo "║                                                                      ║"
  echo "║  Most models require a valid Hugging Face token to download.         ║"
  echo "║  Without it, gated models (Llama, Gemma, etc.) will FAIL.            ║"
  echo "║                                                                      ║"
  echo "║  Note: If you are running a public model (e.g. LiquidAI) or have     ║"
  echo "║  already cached the model locally, you can ignore this warning.      ║"
  echo "║                                                                      ║"
  echo "║  How to get your token:                                              ║"
  echo "║                                                                      ║"
  echo "║   1. Create an account  → https://huggingface.co/join                ║"
  echo "║   2. Go to Settings → Access Tokens:                                 ║"
  echo "║      → https://huggingface.co/settings/tokens                        ║"
  echo "║   3. Click 'Create new token'                                        ║"
  echo "║   4. Name: e.g. 'dgx-spark' — Permission: 'Read'                     ║"
  echo "║   5. Copy the token (starts with hf_...)                             ║"
  echo "║   6. Paste it in this script at line ~40:                            ║"
  echo "║      HUGGING_FACE_HUB_TOKEN=\"hf_YourTokenHere\"                     ║"
  echo "║                                                                      ║"
  echo "║  Or set it as an environment variable before running:                ║"
  echo "║      export HUGGING_FACE_HUB_TOKEN=\"hf_YourTokenHere\"              ║"
  echo "║                                                                      ║"
  echo "║  📖 Doc: https://huggingface.co/docs/hub/en/security-tokens          ║"
  echo "║                                                                      ║"
  echo "╚══════════════════════════════════════════════════════════════════════╝"
  echo ""
fi
# ─────────────────────────────────────────────────────────────────────────────

CONTAINER_NAME_OVERRIDE=""
GPUS_DEVICE="all"
MAX_NUM_SEQS_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""
GPU_MEM_UTIL_OVERRIDE=""
PORT="${PORT:-8000}"
WAIT_FOR_HEALTH=true
ARENA_MODE=false
TP_OVERRIDE=""

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-wait)
      WAIT_FOR_HEALTH=false
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --arena)
      ARENA_MODE=true
      shift
      ;;
    --tp)
      TP_OVERRIDE="$2"
      shift 2
      ;;
    --gpus|--device)
      GPUS_DEVICE="$2"
      shift 2
      ;;
    --name)
      CONTAINER_NAME_OVERRIDE="$2"
      shift 2
      ;;
    --max-num-seqs|--max-seqs)
      MAX_NUM_SEQS_OVERRIDE="$2"
      shift 2
      ;;
    --max-model-len|--max-len)
      MAX_MODEL_LEN_OVERRIDE="$2"
      shift 2
      ;;
    --gpu-memory-utilization|--mem-util)
      GPU_MEM_UTIL_OVERRIDE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-wait] [--port <port>] [--model <model>] [--arena] [--tp <tp_size>] [--gpus <gpu_devices>] [--name <container_name>] [--max-seqs <max_num_seqs>] [--max-len <max_model_len>] [--mem-util <gpu_memory_utilization>]"
      exit 1
      ;;
  esac
done

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
# Select the default model to launch. If the MODEL environment variable or the
# --model command-line option is set, it will take precedence.

if [[ -z "${MODEL}" ]]; then
  # Define the models list: MODEL_ID|GB10_COUNT|DESCRIPTION
  MODELS=(
    "AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4|1|#1  Qwen 3.6 35B NVFP4 Heretic + DFlash drafter (~71 tok/s, up to 117 w/ DFlash) - FASTEST"
    "rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm|1|#2  Qwen 3.6 35B PrismaQuant 4.75bit (~59 tok/s, 256K context)"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4|1|#3  Nemotron-3 Nano 30B NVFP4 (~58 tok/s, 256K context, MoE)"
    "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4|1|#4  Gemma-4 26B NVFP4 (~50 tok/s, 262K context, multimodal)"
    "Neural-ICE/Gemma-4-E2B-it-NVFP4|1|──  Gemma-4 E2B Instruct NVFP4 (~120 tok/s, 128K context, multimodal, audio, FP4 quantized)"
    "bg-digitalservices/Gemma-4-E2B-NVFP4|1|──  Gemma-4 E2B Base NVFP4 (~120 tok/s, 128K context, multimodal, audio, FP4 quantized)"
    "Qwen/Qwen3.6-35B-A3B-FP8|1|#5  Qwen 3.6 35B FP8 (~30 tok/s, 256K context)"
    "rdtand/MiniMax-M2.7-PrismaQuant-3.20bit-vllm|1|#6  MiniMax-M2.7 PrismaQuant 3.20bit (~25 tok/s, 196K context)"
    "Intel/Qwen3-Coder-Next-int4-AutoRound|1|#7  Qwen 3 Coder Next int4 AutoRound (~17 tok/s, 1M context)"
    "RedHatAI/Qwen3.5-122B-A10B-NVFP4|2|#8  Qwen 3.5 122B NVFP4 (~17 tok/s, BEST QUALITY, 64K context)"
    "shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC|1|──  Qwen 3.5 122B int4 AutoRound (z-lab DFlash, 196K context)"

    "RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic|1|──  Mistral-Small 24B Instruct v2501 FP8 (highly optimized for single GB10, 32K context)"

    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4|1|#9  Nemotron-3 Super 120B NVFP4 (~15 tok/s, 128K context)"
    "LiquidAI/LFM2.5-350M|1|──  LiquidAI LFM 2.5 350M (~212 tok/s, BF16 dense, ultra-lightweight)"
    "Qwen/Qwen3.5-0.8B|1|──  Qwen 3.5 0.8B (~103 tok/s, BF16 dense, ultra-lightweight)"

    "nvidia/MiniMax-M2.7-NVFP4|4|──  MiniMax-M2.7 NVFP4 (~24 tok/s, TP=4)"
    "cyankiwi/MiniMax-M2.5-AWQ-4bit|4|──  MiniMax-M2.5 AWQ 4bit (TP=4)"
    "cyankiwi/MiniMax-M2.7-AWQ-4bit|2|──  MiniMax-M2.7 AWQ 4bit (TP=2)"

    "nvidia/Kimi-K2.6-NVFP4|8|──  Kimi-K2.6 NVFP4 MoE (TP=8 Ray cluster, drop-caches mod)"
    "deepseek-ai/DeepSeek-V4-Flash|2|──  DeepSeek V4 Flash FP8 (TP=2, 200K context)"
    "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10|4|──  MiniMax-M2.7 REAP 139B NVFP4 (TP=4)"
  )

  # Check if stdout/stdin are TTYs (interactive mode)
  if [[ -t 1 && -t 0 ]]; then
    # Gorgeous color-coded interactive select menu
    CYAN='\033[0;36m'
    BRIGHT_CYAN='\033[1;36m'
    GREEN='\033[0;32m'
    BRIGHT_GREEN='\033[1;32m'
    YELLOW='\033[0;33m'
    BRIGHT_YELLOW='\033[1;33m'
    ORANGE='\033[38;5;208m'
    BRIGHT_ORANGE='\033[1;38;5;208m'
    RED='\033[0;31m'
    BRIGHT_RED='\033[1;31m'
    NC='\033[0m' # No Color
    BOLD='\033[1m'
    
    echo -e "${BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BRIGHT_CYAN}║             🚀  NVIDIA DGX SPARK (GB10) — vLLM LAUNCHER             ║${NC}"
    echo -e "${BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "Unified Grace Blackwell Memory Architecture (128 GB VRAM per GPU)\n"
    echo -e "${BOLD}Please select an AI model to load:${NC}\n"
    
    print_category() {
      local req_gpus="$1"
      local cat_title="$2"
      local cat_color="$3"
      
      echo -e "${cat_color}── ${cat_title} ──────────────────────────────────────────────────${NC}"
      
      for item in "${MODELS[@]}"; do
        IFS='|' read -r model_id gpus desc <<< "${item}"
        if [[ "${gpus}" -eq "${req_gpus}" ]]; then
          # Find overall 1-based index
          local idx=1
          for search_item in "${MODELS[@]}"; do
            if [[ "${search_item}" == "${item}" ]]; then
              break
            fi
            idx=$((idx + 1))
          done
          
          if [[ "${model_id}" == "AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4" ]]; then
            printf "  ${cat_color}[%2d]${NC} %-55s ${BRIGHT_GREEN}← DEFAULT${NC}\n" "${idx}" "${model_id}"
            printf "       ${cat_color}↳${NC} %s\n" "${desc}"
          else
            printf "  ${cat_color}[%2d]${NC} %s\n" "${idx}" "${model_id}"
            printf "       ${cat_color}↳${NC} %s\n" "${desc}"
          fi
        fi
      done
      echo ""
    }
    
    print_category 1 "🟢 x1 GB10 GPU REQUIRED (Fits in 128 GB VRAM)" "${BRIGHT_GREEN}"
    print_category 2 "🟡 x2 GB10 GPUs REQUIRED (Fits in 256 GB VRAM)" "${BRIGHT_YELLOW}"
    print_category 4 "🟠 x4 GB10 GPUs REQUIRED (Fits in 512 GB VRAM)" "${BRIGHT_ORANGE}"
    print_category 8 "🔴 x8 GB10 GPUs REQUIRED (Fits in 1024 GB VRAM - Ray Distributed)" "${BRIGHT_RED}"
    
    while true; do
      read -p "👉 Enter model number [1-${#MODELS[@]}] or press Enter for default (#1): " choice
      if [[ -z "${choice}" ]]; then
        MODEL="AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4"
        selected_gpus=1
        break
      fi
      if [[ "${choice}" =~ ^[0-9]+$ ]] && [[ "${choice}" -ge 1 ]] && [[ "${choice}" -le "${#MODELS[@]}" ]]; then
        selected_item="${MODELS[$((choice - 1))]}"
        IFS='|' read -r MODEL selected_gpus selected_desc <<< "${selected_item}"
        break
      fi
      echo -e "❌ ${RED}Invalid selection. Please enter a number between 1 and ${#MODELS[@]}.${NC}"
    done
    
    echo -e "\n${BRIGHT_GREEN}✔ Selected model: ${BOLD}${MODEL}${NC}"
    echo -e "${BRIGHT_GREEN}✔ Resources required: ${BOLD}${selected_gpus}x GB10 GPU(s)${NC} (Blackwell SM121)\n"
  else
    # Non-interactive fallback
    DEFAULT_MODEL="AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4"
    MODEL="${DEFAULT_MODEL}"
    echo "ℹ Non-interactive shell detected. Falling back to default model: ${MODEL}"
  fi
fi

# ── RESOLVE TP SIZE ──
# If selected_gpus is not resolved yet, try to find it in the MODELS list array
if [[ -z "${selected_gpus}" ]]; then
  for item in "${MODELS[@]}"; do
    IFS='|' read -r m_id m_gpus m_desc <<< "${item}"
    if [[ "${m_id}" == "${MODEL}" ]]; then
      selected_gpus="${m_gpus}"
      break
    fi
  done
fi

# Fallback selected_gpus to 1 if still not resolved
if [[ -z "${selected_gpus}" ]]; then
  selected_gpus=1
fi

# Resolve final TP size using override if specified
TP_SIZE="${TP_OVERRIDE:-${selected_gpus}}"
if [[ -z "${TP_SIZE}" || "${TP_SIZE}" -lt 1 ]]; then
  TP_SIZE=1
fi

echo -e "${BRIGHT_GREEN}✔ Tensor Parallel size: ${BOLD}${TP_SIZE}${NC}\n"

# Resolve container name
if [[ -z "${CONTAINER_NAME_OVERRIDE}" ]]; then
  # Sanitize model name: replace slashes and other non-alphanumeric chars with dashes
  safe_model_name=$(echo "${MODEL}" | tr '/' '-' | tr -cd '[:alnum:]_-')
  CONTAINER_NAME="mix-vllm-${safe_model_name}"
else
  CONTAINER_NAME="${CONTAINER_NAME_OVERRIDE}"
fi

echo -e "${BRIGHT_GREEN}✔ Container name: ${BOLD}${CONTAINER_NAME}${NC}\n"

# ── ARENA MODE SUITE RUNNER ───────────────────────────────────────────────────
if [[ "${ARENA_MODE}" == "true" ]]; then
  echo "======================================================================"
  echo "🏆              STARTING SPARK ARENA BENCHMARK SUITE                  "
  echo "======================================================================"
  echo "Testing the following models sequentially:"
  
  ARENA_MODELS=(
    "LiquidAI/LFM2.5-350M"
    "Qwen/Qwen3.5-0.8B"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    "AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4"
  )
  
  for am in "${ARENA_MODELS[@]}"; do
    echo "  - ${am}"
  done
  echo "======================================================================"
  echo ""

  RESULTS_FILE="arena_benchmark_results.md"
  echo "# 🏆 Spark Arena Benchmark Results" > "${RESULTS_FILE}"
  echo "Generated on $(date)" >> "${RESULTS_FILE}"
  echo "" >> "${RESULTS_FILE}"
  echo "| Model | TTFT Avg | Tokens/s (Req) | Tokens/s (Agg) | Status |" >> "${RESULTS_FILE}"
  echo "| :--- | :---: | :---: | :---: | :---: |" >> "${RESULTS_FILE}"

  for am in "${ARENA_MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "🚀 Starting arena run for model: ${am}"
    echo "----------------------------------------------------------------------"
    
    # Run mix-vllm.sh child command synchronously to boot and wait for health
    ./mix-vllm.sh --port "${PORT}" --model "${am}"
    
    if [[ $? -ne 0 ]]; then
      echo "❌ Failed to start/healthcheck ${am}, skipping..."
      echo "| ${am} | N/A | N/A | N/A | ❌ FAILED |" >> "${RESULTS_FILE}"
      continue
    fi
    
    echo "📊 Running benchmark for ${am}..."
    rm -f arena_temp.json
    python3 benchmark.py --base-url "http://localhost:${PORT}" --num-requests 4 --concurrency 1 -o "arena_temp.json"
    
    if [[ -f "arena_temp.json" ]]; then
      stats=$(python3 -c "
import json
try:
    with open('arena_temp.json') as f:
        data = json.load(f)
    results = data.get('results', [])
    if results:
        res = results[0]
        print(f'{res.get(\"ttft_avg_ms\", 0):.1f}ms|{res.get(\"tps_per_request_avg\", 0):.2f} t/s|{res.get(\"tps_aggregate\", 0):.2f} t/s')
    else:
        print('N/A|N/A|N/A')
except Exception as e:
    print('N/A|N/A|N/A')
")
      IFS='|' read -r r_ttft r_req r_agg <<< "${stats}"
      echo "| ${am} | ${r_ttft} | ${r_req} | ${r_agg} | ✅ SUCCESS |" >> "${RESULTS_FILE}"
      rm -f arena_temp.json
    else
      echo "| ${am} | N/A | N/A | N/A | ⚠️ NO RESULTS |" >> "${RESULTS_FILE}"
    fi

    echo "🛑 Stopping container for ${am}..."
    docker stop "${CONTAINER_NAME}" >/dev/null 2>/dev/null
    docker rm "${CONTAINER_NAME}" >/dev/null 2>/dev/null
    echo ""
  done

  echo "======================================================================"
  echo "🏆              ALL ARENA BENCHMARKS COMPLETED!                       "
  echo "======================================================================"
  cat "${RESULTS_FILE}"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────

# ── KNOWN IMAGES ─────────────────────────────────────────────────────────────
IMG_AEON7="ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2"
#         └─ vLLM HEAD cu130/sm_120 + FlashInfer 0.6.8 + 8 SM121 patches
#            NVFP4 CUTLASS + DFlash speculative decoding (Qwen3.6 only)

IMG_EUGR="ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"
#         └─ Standard Spark Arena image — NVFP4/FP8/AWQ, Nemotron, Qwen3.5/3.6

IMG_NIGHTLY="vllm/vllm-openai:cu130-nightly"
#         └─ Official nightly with CUDA 13.0 + SM121 kernel support
#            Best for FP8 / BF16 models that don't need NVFP4 patches

IMG_STOCK="vllm/vllm-openai:latest"
#         └─ Standard stable image — BF16/FP8, no NVFP4 SM121 patches


IMG_MINIMAX_NVFP4="ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026050601"
#         └─ Custom nightly for MiniMax-M2.7-NVFP4

IMG_VLLM_NODE="vllm-node"
#         └─ Local image for Ray distributed multi-node serving

IMG_DSV4="vllm-node-dsv4"
#         └─ Local image for DeepSeek V4 Flash with PR 41834 SM12x support

IMG_TF5="vllm-node-tf5"
#         └─ Local image for Qwen3.5-122B-A10B-int4-Autoround-EC (built with --tf5)
# ─────────────────────────────────────────────────────────────────────────────

# Defaults (overridden per model)
VLLM_IMAGE="${IMG_STOCK}"
EXTRA_ARGS=()
ENV_ARGS=()
VOLUME_ARGS=()
MODEL_DOWNLOADS=()   # "repo_id|/local/dir" or "repo_id" (HF cache)

case "${MODEL}" in

  # ═══════════════════════════════════════════════════════════════════════════
  # #1 ★ AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4   88–117 tok/s  ← FASTEST
  # ═══════════════════════════════════════════════════════════════════════════
  # Dedicated image: ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
  # 8 SM121 patches, CUTLASS NVFP4 (not Marlin), DFlash spec-decode.
  # DFlash drafter (z-lab/Qwen3.6-35B-A3B-DFlash, ~870 MB) boosts to 117 tok/s.
  # Both models are auto-downloaded to /opt/qwen36/ if not present.
  #
  # ⚠️  DO NOT set max_model_len=262144 on first boot — torch.compile autotune
  #     will freeze the system. Use 131072 (safe) then raise if needed.
  # ═══════════════════════════════════════════════════════════════════════════
  "AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4")
    VLLM_IMAGE="${IMG_AEON7}"
    GPU_MEM_UTIL=0.50
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=4096
    MAX_NUM_SEQS=2

    MODEL_DOWNLOADS=(
      "AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4|/opt/qwen36/model"
      "z-lab/Qwen3.6-35B-A3B-DFlash|/opt/qwen36/dflash"
    )

    VOLUME_ARGS=(
      -v /opt/qwen36/model:/models/qwen36:ro
      -v /opt/qwen36/dflash:/models/qwen36-dflash:ro
    )

    ENV_ARGS=(
      -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      -e TORCH_MATMUL_PRECISION=high
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      -e NVIDIA_FORWARD_COMPAT=1
      -e TORCHINDUCTOR_MAX_AUTOTUNE=0
      -e TRITON_MAX_AUTOTUNE=0
      -e VLLM_USE_V1=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "qwen3.6-35b-aeon7"
      "--dtype"               "auto"
      "--quantization"        "compressed-tensors"
      "--tensor-parallel-size" "${TP_SIZE}"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
      "--speculative-config"  '{"method":"dflash","model":"/models/qwen36-dflash","num_speculative_tokens":15}'
    )
    # Override MODEL for docker run command — local path replaces HF ID
    MODEL="/models/qwen36"
    ;;


  # ═══════════════════════════════════════════════════════════════════════════
  # #3 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4   ~56 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # Best throughput among ready-to-run models (no pre-download needed).
  # MoE: 30B total / 3.5B active. Reasoning + tool-call capable.
  # Uses built-in 'nemotron_v3' reasoning parser (no external plugin needed).
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.70
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_USE_FLASHINFER_MOE_FP4=1
      -e VLLM_FLASHINFER_MOE_BACKEND=throughput
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "nemotron-nano-30b"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--kv-cache-dtype"      "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "nemotron_v3"
    )
    ;;


  # ═══════════════════════════════════════════════════════════════════════════
  # #5 Qwen/Qwen3.6-35B-A3B-FP8   ~28–30 tok/s single / 156 tok/s aggregate
  # ═══════════════════════════════════════════════════════════════════════════
  # Solid FP8 baseline. cu130-nightly already includes SM121 kernels compiled.
  # Great aggregate throughput under concurrent load (c=32 → 156 tok/s).
  # No NVFP4 patches needed — standard nightly is near-optimal for FP8 here.
  # ═══════════════════════════════════════════════════════════════════════════
  "Qwen/Qwen3.6-35B-A3B-FP8")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=64

    ENV_ARGS=(
      -e VLLM_FLASHINFER_MOE_BACKEND=latency
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "qwen3.6-35b-fp8"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #6 nvidia/Nemotron-3-Super-120B-A12B-NVFP4   ~20–22 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # Model ~70 GB + KV cache ~34 GB → fits in 128 GB.
  # Marlin dequant (FP4→BF16) — native FP4 compute not yet on SM121 in vLLM.
  # Uses built-in 'nemotron_v3' reasoning parser (no external plugin needed).
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.75
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "nemotron-super-120b"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--kv-cache-dtype"      "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "nemotron_v3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #7 RedHatAI/Qwen3.5-122B-A10B-NVFP4   ~16–17 tok/s  ← BEST QUALITY
  # ═══════════════════════════════════════════════════════════════════════════
  # Best output quality on single Spark — RedHatAI calibration ≈ FP16 quality.
  # Model ~75 GB; gpu_mem_util=0.90 required to leave room for KV cache.
  # ═══════════════════════════════════════════════════════════════════════════
  "RedHatAI/Qwen3.5-122B-A10B-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.90
    MAX_MODEL_LEN=65536
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "qwen3.5-122b"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--attention-backend"   "flashinfer"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Qwen3.5-122B-A10B-int4-AutoRound-EC + z-lab DFlash, 196K context, 16K prefill
  # ═══════════════════════════════════════════════════════════════════════════
  "shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC" | "Qwen3.5-122B-A10B-int4-Autoround-EC-DFLASH-FLashQLA-SlidingWindowAttention")
    VLLM_IMAGE="${IMG_TF5}"
    GPU_MEM_UTIL=0.82
    MAX_MODEL_LEN=196608
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=8

    MODEL_DOWNLOADS=(
      "shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC"
      "z-lab/Qwen3.5-122B-A10B-DFlash"
    )

    ENV_ARGS=(
      -e VLLM_ENABLE_CUDAGRAPH_GC=1
      -e FLASHINFER_DISABLE_VERSION_CHECK=1
      -e VLLM_USE_FLASHINFER_SAMPLER=1
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e TZ=Europe/Vienna
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"           "qwen"
      "--dtype"                       "bfloat16"
      "--load-format"                 "fastsafetensors"
      "--attention-backend"           "flash_attn"
      "--speculative-config"          '{"method":"dflash","model":"z-lab/Qwen3.5-122B-A10B-DFlash","num_speculative_tokens":5}'
      "--enable-prompt-tokens-details"
      "--enable-auto-tool-choice"
      "--tool-call-parser"            "qwen3_coder"
      "--chat-template"               "/qwen3.5-enhanced.jinja"
      "--reasoning-parser"            "qwen3"
      "--generation-config"           "auto"
      "--override-generation-config"  '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}'
    )

    # Override MODEL for docker run command if full name was used
    MODEL="shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC"
    ;;



  # ═══════════════════════════════════════════════════════════════════════════
  # Gemma 4 26B A4B — NVFP4 (bg-digitalservices), TP=4, 262k context
  # ⚠️  Requires fix-gemma4-tool-parser — present in IMG_EUGR (recent builds)
  # ═══════════════════════════════════════════════════════════════════════════
  "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"    "gemma4-26b"
      "--dtype"                "auto"
      "--load-format"          "fastsafetensors"
      "--kv-cache-dtype"       "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
      "--enable-auto-tool-choice"
      "--tool-call-parser"     "gemma4"
      "--reasoning-parser"     "gemma4"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Gemma 4 E2B Instruct NVFP4 — Neural-ICE, NVFP4 Quantized E2B-it, 128k context
  # ⚠️  Requires fix-gemma4-tool-parser — present in IMG_EUGR (recent builds)
  # ═══════════════════════════════════════════════════════════════════════════
  "Neural-ICE/Gemma-4-E2B-it-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=16

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"    "gemma4-e2b-it-nvfp4"
      "--dtype"                "auto"
      "--load-format"          "fastsafetensors"
      "--kv-cache-dtype"       "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
      "--enable-auto-tool-choice"
      "--tool-call-parser"     "gemma4"
      "--reasoning-parser"     "gemma4"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Gemma 4 E2B Base NVFP4 — bg-digitalservices, NVFP4 Quantized E2B (Pre-trained), 128k context
  # ═══════════════════════════════════════════════════════════════════════════
  "bg-digitalservices/Gemma-4-E2B-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=16

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"    "gemma4-e2b-base-nvfp4"
      "--dtype"                "auto"
      "--load-format"          "fastsafetensors"
      "--kv-cache-dtype"       "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Qwen3.6 35B A3B — PrismaQuant 4.75bit + MTP speculative + 256k context
  # ═══════════════════════════════════════════════════════════════════════════
  "rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm")
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.60
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e VLLM_TUNED_CONFIG_FOLDER=/workspace/moe-configs
      -e FLASHINFER_DISABLE_VERSION_CHECK=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "qwen3.6-35b-prisma"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--attention-backend"   "flashinfer"
      "--quantization"        "compressed-tensors"
      "--kv-cache-dtype"      "fp8"
      "--reasoning-parser"    "qwen3"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--optimization-level"  "3"
      "--performance-mode"    "throughput"
      "--default-chat-template-kwargs" '{"preserve_thinking":true}'
      "--speculative-config"  '{"method":"mtp","num_speculative_tokens":3}'
      "--override-generation-config" '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.1,"repetition_penalty":1.01}'
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Qwen3 Coder Next — int4 AutoRound (Intel), 1M context, MoE FP8
  # ═══════════════════════════════════════════════════════════════════════════
  "Intel/Qwen3-Coder-Next-int4-AutoRound")
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.70
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=49152
    MAX_NUM_SEQS=384

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_SLEEP_WHEN_IDLE=0
      -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
      -e SAFETENSORS_FAST_GPU=1
      -e VLLM_USE_FLASHINFER_MOE_FP8=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "qwen3-coder-next"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
      "--load-format"         "fastsafetensors"
      "--language-model-only"
      "--kv-cache-dtype"      "fp8"
      "--optimization-level"  "3"
      "--performance-mode"    "throughput"
      "--mamba-cache-mode"    "align"
      "--hf-overrides" '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}'
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # LiquidAI LFM2.5 350M — ultra-lightweight, for testing/development
  # ═══════════════════════════════════════════════════════════════════════════
  "LiquidAI/LFM2.5-350M")
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "lfm2.5-350m"
      "--language-model-only"
      "--load-format"         "fastsafetensors"
      "--attention-backend"   "flash_attn"
      "--dtype"               "auto"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Qwen 3.5 0.8B — Lightweight dense model, extremely fast, 262k context
  # ═══════════════════════════════════════════════════════════════════════════
  "Qwen/Qwen3.5-0.8B")
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "qwen3.5-0.8b"
      "--language-model-only"
      "--load-format"         "fastsafetensors"
      "--attention-backend"   "flash_attn"
      "--dtype"               "auto"
      "--reasoning-parser"    "qwen3"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
    )
    ;;



  # ═══════════════════════════════════════════════════════════════════════════
  # nvidia/MiniMax-M2.7-NVFP4
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/MiniMax-M2.7-NVFP4")
    VLLM_IMAGE="${IMG_MINIMAX_NVFP4}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"           "minimax-m2.7-nvfp4"
      "-tp"                           "${TP_SIZE}"
      "-pp"                           "1"
      "--load-format"                 "instanttensor"
      "--enable-auto-tool-choice"
      "--tool-call-parser"            "minimax_m2"
      "--reasoning-parser"            "minimax_m2"
      "--kv-cache-dtype"              "fp8"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # cyankiwi/MiniMax-M2.5-AWQ-4bit
  # ═══════════════════════════════════════════════════════════════════════════
  "cyankiwi/MiniMax-M2.5-AWQ-4bit")
    VLLM_IMAGE="${IMG_VLLM_NODE}"
    GPU_MEM_UTIL=0.7
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"            "minimax-m2.5-awq"
      "-tp"                            "${TP_SIZE}"
      "--distributed-executor-backend" "ray"
      "--load-format"                  "fastsafetensors"
      "--enable-auto-tool-choice"
      "--tool-call-parser"             "minimax_m2"
      "--reasoning-parser"             "minimax_m2"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # cyankiwi/MiniMax-M2.7-AWQ-4bit
  # ═══════════════════════════════════════════════════════════════════════════
  "cyankiwi/MiniMax-M2.7-AWQ-4bit")
    VLLM_IMAGE="${IMG_VLLM_NODE}"
    GPU_MEM_UTIL=0.7
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"            "minimax-m2.7-awq"
      "-tp"                            "${TP_SIZE}"
      "--distributed-executor-backend" "ray"
      "--load-format"                  "fastsafetensors"
      "--enable-auto-tool-choice"
      "--tool-call-parser"             "minimax_m2"
      "--reasoning-parser"             "minimax_m2"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # nvidia/Kimi-K2.6-NVFP4 on 8 DGX Spark nodes
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/Kimi-K2.6-NVFP4")
    VLLM_IMAGE="${IMG_VLLM_NODE}"
    GPU_MEM_UTIL=0.72
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=2048
    MAX_NUM_SEQS=1
    MODEL_DOWNLOADS=("none")

    VOLUME_ARGS=(
      -v /mnt/glm51-hf-cache:/root/.cache/huggingface
      -v /mnt/glm51-hf-cache:/mnt/glm51-hf-cache
    )

    ENV_ARGS=(
      -e HF_HUB_OFFLINE=1
      -e TRANSFORMERS_OFFLINE=1
      -e HF_HOME=/mnt/glm51-hf-cache
      -e VLLM_DISTRIBUTED_EXECUTOR_CONFIG='{"placement_group_options":{"strategy":"SPREAD"}}'
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_USE_FLASHINFER_SAMPLER=0
      -e OMP_NUM_THREADS=4
    )

    EXTRA_ARGS=(
      "--tensor-parallel-size"         "${TP_SIZE}"
      "--distributed-executor-backend" "ray"
      "--enforce-eager"
      "--kv-cache-dtype"              "fp8"
      "--mm-processor-cache-gb"       "0"
      "--tool-call-parser"            "kimi_k2"
      "--reasoning-parser"            "kimi_k2"
      "--enable-auto-tool-choice"
      "--served-model-name"           "kimi-k2.6-nvfp4"
      "--no-async-scheduling"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # deepseek-ai/DeepSeek-V4-Flash on dual DGX Spark TP=2
  # ═══════════════════════════════════════════════════════════════════════════
  "deepseek-ai/DeepSeek-V4-Flash")
    VLLM_IMAGE="${IMG_DSV4}"
    GPU_MEM_UTIL=0.9
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=4192
    MAX_NUM_SEQS=20

    ENV_ARGS=(
      -e TORCH_CUDA_ARCH_LIST=12.1a
      -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      -e VLLM_TRITON_MLA_SPARSE=1
      -e FLASHINFER_DISABLE_VERSION_CHECK=1
      -e TILELANG_CLEANUP_TEMP_FILES=1
      -e DG_JIT_USE_NVRTC=0
      -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
      -e DG_JIT_PRINT_COMPILER_COMMAND=1
      -e NCCL_IB_DISABLE=0
      -e NCCL_DEBUG=WARN
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"            "deepseek-v4-flash"
      "-tp"                            "${TP_SIZE}"
      "-pp"                           "1"
      "--kv-cache-dtype"               "fp8"
      "--block-size"                   "256"
      "--distributed-executor-backend" "mp"
      "--compilation-config"           '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
      "--speculative-config"           '{"method":"mtp","num_speculative_tokens":2}'
      "--tokenizer-mode"               "deepseek_v4"
      "--tool-call-parser"             "deepseek_v4"
      "--enable-auto-tool-choice"
      "--reasoning-parser"             "deepseek_v4"
      "--reasoning-config"             '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
      "--default-chat-template-kwargs" '{"thinking":true}'
      "--load-format"                  "safetensors"
    )
    ;;





  # ═══════════════════════════════════════════════════════════════════════════
  # RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic — Optimized FP8, 32K context
  # ═══════════════════════════════════════════════════════════════════════════
  "RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=16

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"    "mistral-small-24b"
      "--dtype"                "auto"
      "--load-format"          "fastsafetensors"
      "--quantization"         "compressed-tensors"
      "--kv-cache-dtype"       "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # MiniMax-M2.7-PrismaQuant-3.20bit-vllm (rdtand), 196K context, 3.20bit
  # ═══════════════════════════════════════════════════════════════════════════
  "rdtand/MiniMax-M2.7-PrismaQuant-3.20bit-vllm")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.90
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"    "minimax-m2.7-prisma"
      "--dtype"                "auto"
      "--load-format"          "safetensors"
      "--quantization"         "compressed-tensors"
      "--kv-cache-dtype"       "fp8"
      "--tensor-parallel-size" "${TP_SIZE}"
      "--enable-auto-tool-choice"
      "--tool-call-parser"     "minimax_m2"
      "--reasoning-parser"     "minimax_m2"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10, TP=4, NVFP4 Quant, GB10
  # ═══════════════════════════════════════════════════════════════════════════
  "dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10")
    VLLM_IMAGE="${IMG_MINIMAX_NVFP4}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=102400
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"           "minimax-m2.7-reap-139b"
      "-tp"                           "${TP_SIZE}"
      "-pp"                           "1"
      "--load-format"                 "instanttensor"
      "--enable-auto-tool-choice"
      "--tool-call-parser"            "minimax_m2"
      "--reasoning-parser"            "minimax_m2"
      "--kv-cache-dtype"              "fp8"
    )
    ;;


  *)
    echo -e "\033[1;33m⚠️  WARNING — Custom model '${MODEL}' is not in the pre-configured catalog!\033[0m"
    echo -e "\033[1;33m  Using general stock settings: stock image, 32K context, 80% memory utilization.\033[0m\n"
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=128
    ;;
esac

# Apply overrides if specified
if [[ -n "${MAX_NUM_SEQS_OVERRIDE}" ]]; then
  MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE}"
fi
if [[ -n "${MAX_MODEL_LEN_OVERRIDE}" ]]; then
  MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE}"
fi
if [[ -n "${GPU_MEM_UTIL_OVERRIDE}" ]]; then
  GPU_MEM_UTIL="${GPU_MEM_UTIL_OVERRIDE}"
fi

# Force disable the unstable experimental V1 engine unless explicitly enabled by AEON-7
if [[ "${MODEL}" != "AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4" ]]; then
  ENV_ARGS+=(-e VLLM_USE_V1=0)
fi

# ── AUTO-DOWNLOAD MODELS ─────────────────────────────────────────────────────
# Downloads models that are not yet present locally.
# Uses Docker + Python huggingface_hub (always present in vLLM images).
#   - "repo_id|/local/dir"  → downloads to a specific directory (for volume mounts)
#   - "repo_id"             → downloads to the HF cache (~/.cache/huggingface)
# If MODEL_DOWNLOADS is empty, defaults to caching ${MODEL} from HuggingFace.
# ─────────────────────────────────────────────────────────────────────────────

download_if_needed() {
  local entry="$1"
  local repo="${entry%%|*}"
  local target_dir="${entry#*|}"

  local dl_image="${VLLM_IMAGE}"
  # Fall back to stable stock image for downloading if the image is a custom local-only one
  if [[ "${dl_image}" == "vllm-node"* || "${dl_image}" == "vllm-glm51"* ]]; then
    dl_image="${IMG_STOCK}"
  fi

  # If no | separator, target_dir equals repo → cache mode
  if [[ "${target_dir}" == "${repo}" ]]; then
    target_dir=""
  fi

  if [[ -n "${target_dir}" ]]; then
    # ── Local directory download ──
    if [[ ! -f "${target_dir}/config.json" && ! -f "${target_dir}/params.json" ]]; then
      echo "📥 Downloading ${repo} → ${target_dir} ..."
      echo "⏳ Note: With a good internet connection, it takes on average 10 minutes to download/cache a model."
      mkdir -p "${target_dir}"
      docker run --rm \
        --entrypoint python3 \
        -v "${target_dir}:${target_dir}" \
        -e HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
        "${dl_image}" \
        -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}', local_dir='${target_dir}')"
      if [[ $? -ne 0 ]]; then
        echo "❌ Failed to download ${repo}"
        echo "💡 If this is a gated model (like Gemma 3), ensure you accepted the license terms on Hugging Face:"
        echo "   👉 https://huggingface.co/${repo}"
        exit 1
      fi
    else
      echo "✓  ${repo} → ${target_dir} (already present)"
    fi

    # ── PATCH FOR NVFP4_AWQ in target_dir (via Docker for root permissions) ──
    docker run --rm \
      -v "${target_dir}:${target_dir}" \
      --entrypoint python3 \
      "${dl_image}" \
      -c "
import os, glob
target_dir = '${target_dir}'
for pattern in ['**/config.json', '**/hf_quant_config.json']:
    for filepath in glob.glob(os.path.join(target_dir, pattern), recursive=True):
        real_path = os.path.realpath(filepath)
        if os.path.exists(real_path):
            with open(real_path, 'r') as f:
                content = f.read()
            if '\"quant_algo\": \"NVFP4_AWQ\"' in content:
                print('   Found quant_algo: NVFP4_AWQ in ' + real_path + '. Patching...')
                os.chmod(real_path, 0o666)
                with open(real_path, 'w') as f:
                    f.write(content.replace('\"quant_algo\": \"NVFP4_AWQ\"', '\"quant_algo\": \"NVFP4\"'))
"
  else
    # ── HuggingFace cache download ──
    local cache_name="models--$(echo "${repo}" | tr '/' '--')"
    local cache_path="${HOME}/.cache/huggingface/hub/${cache_name}"
    if [[ ! -d "${cache_path}/snapshots" ]]; then
      echo "📥 Caching ${repo} ..."
      echo "⏳ Note: With a good internet connection, it takes on average 10 minutes to download/cache a model."
      docker run --rm \
        --entrypoint python3 \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -e HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
        "${dl_image}" \
        -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}')"
      if [[ $? -ne 0 ]]; then
        echo "❌ Failed to download ${repo}"
        echo "💡 If this is a gated model (like Gemma 3), ensure you accepted the license terms on Hugging Face:"
        echo "   👉 https://huggingface.co/${repo}"
        exit 1
      fi
    else
      echo "✓  ${repo} (already cached)"
    fi

    # ── PATCH FOR NVFP4_AWQ in cache (via Docker for root permissions) ──
    if [[ -d "${cache_path}/snapshots" ]]; then
      echo "🔧 Checking/patching quantization config in HF cache for ${repo} via Docker..."
      docker run --rm \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        --entrypoint python3 \
        "${dl_image}" \
        -c "
import os, glob
cache_dir = '/root/.cache/huggingface/hub/${cache_name}/snapshots'
for pattern in ['**/config.json', '**/hf_quant_config.json']:
    for filepath in glob.glob(os.path.join(cache_dir, pattern), recursive=True):
        real_path = os.path.realpath(filepath)
        if os.path.exists(real_path):
            with open(real_path, 'r') as f:
                content = f.read()
            if '\"quant_algo\": \"NVFP4_AWQ\"' in content:
                print('   Found quant_algo: NVFP4_AWQ in ' + real_path + '. Patching to NVFP4...')
                os.chmod(real_path, 0o666)
                with open(real_path, 'w') as f:
                    f.write(content.replace('\"quant_algo\": \"NVFP4_AWQ\"', '\"quant_algo\": \"NVFP4\"'))
"
    fi
  fi
}

# Default: download the MODEL itself from HF if no explicit downloads defined
if [[ ${#MODEL_DOWNLOADS[@]} -eq 0 ]]; then
  MODEL_DOWNLOADS=("${MODEL}")
fi

# Ensure the local HuggingFace cache folder exists so it belongs to the current user,
# avoiding Docker creating it as root and causing permission issues.
mkdir -p "${HOME}/.cache/huggingface"

for dl in "${MODEL_DOWNLOADS[@]}"; do
  if [[ "${dl}" == "none" ]]; then
    continue
  fi
  download_if_needed "${dl}"
done

# ─────────────────────────────────────────────────────────────────────────────

echo "🔥 ${MODEL}"
echo "   image  : ${VLLM_IMAGE}"
echo "   ctx    : ${MAX_MODEL_LEN}   mem: ${GPU_MEM_UTIL}   seqs: ${MAX_NUM_SEQS}"
echo "   gpus   : ${GPUS_DEVICE}   container: ${CONTAINER_NAME}"

# ── MOD: DROP CACHES ──────────────────────────────────────────────────────────
# If selected model includes drop-caches mod, drop system caches to reclaim RAM
if [[ "${MODEL}" == "nvidia/Kimi-K2.6-NVFP4" ]]; then
  echo "🧹 Applying mod: mods/drop-caches..."
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo "   Writing to /proc/sys/vm/drop_caches..."
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  else
    echo "   Note: Could not drop caches (insufficient permissions), proceeding..."
  fi
fi

# ── ENGINE VERSION SELECTION ──────────────────────────────────────────────────
# Recent vLLM nightlies enable the experimental V1 engine by default.
# However, many of our quantized or custom configurations require the stable V0 engine.
# We default to VLLM_USE_V1=0 unless explicitly requested by the model config.
has_v1_env=false
for env_arg in "${ENV_ARGS[@]}"; do
  if [[ "${env_arg}" == *"VLLM_USE_V1"* ]]; then
    has_v1_env=true
    break
  fi
done
if [[ "${has_v1_env}" == "false" ]]; then
  ENV_ARGS+=(-e VLLM_USE_V1=0)
fi

# Check if the requested custom local image exists before running
if [[ "${VLLM_IMAGE}" == "vllm-node"* || "${VLLM_IMAGE}" == "vllm-glm51"* ]]; then
  if [[ -z "$(docker images -q "${VLLM_IMAGE}")" ]]; then
    echo -e "\033[1;31m❌ ERROR: Local docker image '${VLLM_IMAGE}' is not present!\033[0m"
    echo -e "\033[1;33m💡 This model requires a custom local image that must be built first.\033[0m"
    echo -e "\033[1;33m   Please ensure you have built or imported the '${VLLM_IMAGE}' image before launching this model.\033[0m"
    exit 1
  fi
fi

# Append tensor parallel size if not already specified in EXTRA_ARGS
has_tp_arg=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--tensor-parallel-size" || "${arg}" == "-tp" ]]; then
    has_tp_arg=true
    break
  fi
done

if [[ "${has_tp_arg}" == "false" ]]; then
  EXTRA_ARGS+=("--tensor-parallel-size" "${TP_SIZE}")
fi

docker stop "${CONTAINER_NAME}" 2>/dev/null
docker rm   "${CONTAINER_NAME}" 2>/dev/null

docker run -d \
  --name    "${CONTAINER_NAME}" \
  --gpus    "${GPUS_DEVICE}" \
  --ipc     host \
  --ulimit  memlock=-1 \
  --ulimit  stack=67108864 \
  --shm-size 32g \
  -p "${PORT}:8000" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${VOLUME_ARGS[@]}" \
  --restart unless-stopped \
  --entrypoint vllm \
  "${ENV_ARGS[@]}" \
  "${VLLM_IMAGE}" \
  serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len          "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --max-num-seqs           "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"

if [[ "${WAIT_FOR_HEALTH}" == "true" ]]; then
  echo "⏳ Waiting for vLLM server to start and become healthy..."
  echo "   (This can take a few minutes for larger models to load and compile kernels)"
  
  timeout=300
  interval=5
  elapsed=0
  healthy=false
  
  while [[ $elapsed -lt $timeout ]]; do
    # Check if the container is still running
    if ! docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" | grep -q "${CONTAINER_NAME}"; then
      echo ""
      echo "❌ ERROR: Container ${CONTAINER_NAME} stopped running!"
      echo "📋 Showing last 20 lines of container logs to diagnose:"
      echo "──────────────────────────────────────────────────────────────────────────────"
      docker logs --tail 20 "${CONTAINER_NAME}"
      echo "──────────────────────────────────────────────────────────────────────────────"
      exit 1
    fi
    
    # Check vLLM health endpoint
    if curl -s "http://localhost:${PORT}/health" | grep -q "ok" || curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" | grep -q "200"; then
      healthy=true
      break
    fi
    
    # Print progress dot and sleep
    printf "."
    sleep $interval
    elapsed=$((elapsed + interval))
  done
  
  echo ""
  if [[ "${healthy}" == "true" ]]; then
    echo "🚀 Server is HEALTHY and ready to serve requests!"
  else
    echo "❌ TIMEOUT: Server did not become healthy within ${timeout} seconds."
    echo "📋 Showing last 20 lines of container logs to diagnose:"
    echo "──────────────────────────────────────────────────────────────────────────────"
    docker logs --tail 20 "${CONTAINER_NAME}"
    echo "──────────────────────────────────────────────────────────────────────────────"
    exit 1
  fi
fi

echo "✅ ${CONTAINER_NAME} → http://localhost:${PORT}"
echo "📋 Logs   : docker logs -f ${CONTAINER_NAME}"
echo "🔍 Health : curl http://localhost:${PORT}/health"
echo "📊 Test   : python3 benchmark.py --base-url http://localhost:${PORT}"
