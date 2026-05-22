#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# mix-vllm.sh — vLLM launcher for DGX Spark (GB10 / SM121, 128 GB)
# Models ranked by single-node throughput (Spark Arena leaderboard)
#
#  Rank  Model                                         tok/s    Quant     Image
#  ────  ──────────────────────────────────────────   ───────  ───────   ────────────
#  #1    AEON-7/Qwen3.6-35B-heretic-NVFP4 + DFlash    88–117   NVFP4    aeon-7 v1.2
#  #2    openai/gpt-oss-120b (MXFP4)                   ~60     MXFP4    eugr-nightly
#  #3    nvidia/Nemotron-3-Nano-30B-A3B-NVFP4          ~56     NVFP4    eugr-nightly
#  #4    Qwen/Qwen3.6-35B-A3B-FP8                      ~30     FP8      cu130-nightly
#  #5    nvidia/Nemotron-3-Super-120B-A12B-NVFP4       ~22     NVFP4    eugr-nightly
#  #6    RedHatAI/Qwen3.5-122B-A10B-NVFP4              ~17     NVFP4    eugr-nightly  ← best quality
#  ──    bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4   ~20     NVFP4    eugr-nightly
#  ──    rdtand/Qwen3.6-35B-A3B-PrismaQuant            ~40     4.75bit  vllm-latest
#  ──    Intel/Qwen3-Coder-Next-int4-AutoRound         ~30     INT4     vllm-latest
#  ──    LiquidAI/LFM2.5-350M                         fast     BF16     vllm-latest
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
  echo "║  ⚠️  WARNING — Hugging Face token is NOT configured!               ║"
  echo "╠══════════════════════════════════════════════════════════════════════╣"
  echo "║                                                                    ║"
  echo "║  Most models require a valid Hugging Face token to download.       ║"
  echo "║  Without it, gated models (Llama, Gemma, etc.) will FAIL.          ║"
  echo "║                                                                    ║"
  echo "║  Note: If you are running a public model (e.g. LiquidAI) or have   ║"
  echo "║  already cached the model locally, you can ignore this warning.    ║"
  echo "║                                                                    ║"
  echo "║  How to get your token:                                            ║"
  echo "║                                                                    ║"
  echo "║   1. Create an account  → https://huggingface.co/join              ║"
  echo "║   2. Go to Settings → Access Tokens:                               ║"
  echo "║      → https://huggingface.co/settings/tokens                      ║"
  echo "║   3. Click 'Create new token'                                      ║"
  echo "║   4. Name: e.g. 'dgx-spark' — Permission: 'Read'                  ║"
  echo "║   5. Copy the token (starts with hf_...)                           ║"
  echo "║   6. Paste it in this script at line ~40:                          ║"
  echo "║      HUGGING_FACE_HUB_TOKEN=\"hf_YourTokenHere\"                     ║"
  echo "║                                                                    ║"
  echo "║  Or set it as an environment variable before running:              ║"
  echo "║      export HUGGING_FACE_HUB_TOKEN=\"hf_YourTokenHere\"              ║"
  echo "║                                                                    ║"
  echo "║  📖 Doc: https://huggingface.co/docs/hub/en/security-tokens        ║"
  echo "║                                                                    ║"
  echo "╚══════════════════════════════════════════════════════════════════════╝"
  echo ""
fi
# ─────────────────────────────────────────────────────────────────────────────

CONTAINER_NAME="mix-vllm"
PORT="${PORT:-8000}"
WAIT_FOR_HEALTH=true

ARENA_MODE=false

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
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--no-wait] [--port <port>] [--model <model>] [--arena]"
      exit 1
      ;;
  esac
done

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
# Select the default model to launch. If the MODEL environment variable or the
# --model command-line option is set, it will take precedence.
# To change the default, uncomment ONE of the DEFAULT_MODEL lines below:
#DEFAULT_MODEL="AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4"          # https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4
#DEFAULT_MODEL="openai/gpt-oss-120b"                            # https://huggingface.co/openai/gpt-oss-120b
# DEFAULT_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"   # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
#DEFAULT_MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
#DEFAULT_MODEL="RedHatAI/Qwen3.5-122B-A10B-NVFP4"            # https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4
# DEFAULT_MODEL="bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4"   # https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
# DEFAULT_MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm" # https://huggingface.co/rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm
# DEFAULT_MODEL="Intel/Qwen3-Coder-Next-int4-AutoRound"        # https://huggingface.co/Intel/Qwen3-Coder-Next-int4-AutoRound
# DEFAULT_MODEL="LiquidAI/LFM2.5-350M"                         # https://huggingface.co/LiquidAI/LFM2.5-350M
# DEFAULT_MODEL="Qwen/Qwen3.5-0.8B"                            # https://huggingface.co/Qwen/Qwen3.5-0.8B
#DEFAULT_MODEL="casperhansen/deepseek-r1-distill-qwen-32b-awq" # https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-32b-awq
#DEFAULT_MODEL="nm-testing/DeepSeek-R1-Distill-Qwen-32B-NVFP4" # https://huggingface.co/nm-testing/DeepSeek-R1-Distill-Qwen-32B-NVFP4
#DEFAULT_MODEL="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-FP8"   # https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Qwen-14B-FP8
#DEFAULT_MODEL="casperhansen/deepseek-r1-distill-qwen-14b-awq" # https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-14b-awq
#DEFAULT_MODEL="neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8"   # https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8
#DEFAULT_MODEL="casperhansen/deepseek-r1-distill-llama-8b-awq" # https://huggingface.co/casperhansen/deepseek-r1-distill-llama-8b-awq
#DEFAULT_MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4" # https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
#DEFAULT_MODEL="zai-org/GLM-5.1-FP8"                            # https://huggingface.co/zai-org/GLM-5.1-FP8
#DEFAULT_MODEL="nvidia/MiniMax-M2.7-NVFP4"                      # https://huggingface.co/nvidia/MiniMax-M2.7-NVFP4
#DEFAULT_MODEL="cyankiwi/MiniMax-M2.5-AWQ-4bit"                  # https://huggingface.co/cyankiwi/MiniMax-M2.5-AWQ-4bit
#DEFAULT_MODEL="cyankiwi/MiniMax-M2.7-AWQ-4bit"                  # https://huggingface.co/cyankiwi/MiniMax-M2.7-AWQ-4bit
#DEFAULT_MODEL="nvidia/Kimi-K2.6-NVFP4"                          # https://huggingface.co/nvidia/Kimi-K2.6-NVFP4
#DEFAULT_MODEL="deepseek-ai/DeepSeek-V4-Flash"                   # https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

MODEL="${MODEL:-${DEFAULT_MODEL}}"

# ── ARENA MODE SUITE RUNNER ───────────────────────────────────────────────────
if [[ "${ARENA_MODE}" == "true" ]]; then
  echo "======================================================================"
  echo "🏆              STARTING SPARK ARENA BENCHMARK SUITE                  "
  echo "======================================================================"
  echo "Testing the following models sequentially:"
  
  ARENA_MODELS=(
    "LiquidAI/LFM2.5-350M"
    "Qwen/Qwen3.5-0.8B"
    "neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
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


IMG_EUGR_MXFP4="${IMG_EUGR}"
#         └─ Uses eugr-nightly with CUTLASS MXFP4 backend
#            No local build needed — MXFP4 kernels included in eugr-nightly

IMG_NIGHTLY="vllm/vllm-openai:cu130-nightly"
#         └─ Official nightly with CUDA 13.0 + SM121 kernel support
#            Best for FP8 / BF16 models that don't need NVFP4 patches

IMG_STOCK="vllm/vllm-openai:latest"
#         └─ Standard stable image — BF16/FP8, no NVFP4 SM121 patches

IMG_NEMOTRON_OMNI="vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404"
#         └─ Custom Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 optimized image

IMG_GLM51="vllm-glm51-cu130"
#         └─ Z.AI GLM-5.1-FP8 optimized image with CUDA 13.0 (8-node Ray cluster)

IMG_MINIMAX_NVFP4="ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026050601"
#         └─ Custom nightly for MiniMax-M2.7-NVFP4

IMG_VLLM_NODE="vllm-node"
#         └─ Local image for Ray distributed multi-node serving

IMG_DSV4="vllm-node-dsv4"
#         └─ Local image for DeepSeek V4 Flash with PR 41834 SM12x support
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
    MAX_MODEL_LEN=131072
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
      "--tensor-parallel-size" "1"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
      "--speculative-config"  '{"method":"dflash","model":"/models/qwen36-dflash","num_speculative_tokens":15}'
    )
    # Override MODEL for docker run command — local path replaces HF ID
    MODEL="/models/qwen36"
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #2 openai/gpt-oss-120b (MXFP4)   ~58–60 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # Uses eugr-nightly with CUTLASS MXFP4 backend — no local build needed.
  # ⚠️  SM121 Marlin MXFP4 bug (vllm#37030) causes null output on older builds.
  #     The CUTLASS path (--mxfp4-backend CUTLASS) avoids this.
  # ═══════════════════════════════════════════════════════════════════════════
  "openai/gpt-oss-120b")
    VLLM_IMAGE="${IMG_EUGR_MXFP4}"
    GPU_MEM_UTIL=0.70
    MAX_MODEL_LEN=65536
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "gpt-oss-120b"
      "--dtype"               "bfloat16"
      "--quantization"        "mxfp4"
      "--mxfp4-backend"       "CUTLASS"
      "--mxfp4-layers"        "moe,qkv,o,lm_head"
      "--attention-backend"   "FLASHINFER"
      "--kv-cache-dtype"      "fp8"
      "--load-format"         "fastsafetensors"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "openai"
      "--reasoning-parser"    "openai_gptoss"
    )
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
    MAX_MODEL_LEN=262144
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
      "--tensor-parallel-size" "1"
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
    MAX_MODEL_LEN=262144
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
    MAX_MODEL_LEN=131072
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
      "--tensor-parallel-size" "1"
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
  # Gemma 4 26B A4B — NVFP4 (bg-digitalservices), TP=4, 262k context
  # ⚠️  Requires fix-gemma4-tool-parser — present in IMG_EUGR (recent builds)
  # ═══════════════════════════════════════════════════════════════════════════
  "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=262144
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
      "--tensor-parallel-size" "1"
      "--enable-auto-tool-choice"
      "--tool-call-parser"     "gemma4"
      "--reasoning-parser"     "gemma4"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Qwen3.6 35B A3B — PrismaQuant 4.75bit + MTP speculative + 256k context
  # ═══════════════════════════════════════════════════════════════════════════
  "rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm")
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.60
    MAX_MODEL_LEN=262144
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
    MAX_MODEL_LEN=1048576
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
    MAX_MODEL_LEN=262144
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
  # DeepSeek R1 Distill Qwen 32B AWQ — 32B reasoning model, AWQ, fast on consumer GPUs
  # ═══════════════════════════════════════════════════════════════════════════
  "casperhansen/deepseek-r1-distill-qwen-32b-awq")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "deepseek-r1-qwen-32b-awq"
      "--dtype"               "float16"
      "--load-format"         "fastsafetensors"
      "--quantization"        "awq"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Llama 3.3 70B Instruct FP8 — 70B frontier model, FP8, excellent balance of quality/speed
  # ═══════════════════════════════════════════════════════════════════════════
  "neuralmagic/Llama-3.3-70B-Instruct-FP8")
    VLLM_IMAGE="${IMG_NIGHTLY}"
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
      "--served-model-name"   "llama-3.3-70b-fp8"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--quantization"        "fp8"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "llama3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Llama 3.3 70B Instruct AWQ — 70B frontier model, AWQ, highly compressed for local serving
  # ═══════════════════════════════════════════════════════════════════════════
  "casperhansen/llama-3.3-70b-instruct-awq")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "llama-3.3-70b-awq"
      "--dtype"               "float16"
      "--load-format"         "fastsafetensors"
      "--quantization"        "awq"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "llama3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Llama 3.3 70B Instruct NVFP4 — NVFP4 version of the 70B model, Blackwell optimized
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/Llama-3.3-70B-Instruct-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.90
    MAX_MODEL_LEN=65536
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=4

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "llama-3.3-70b-nvfp4"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--quantization"        "compressed-tensors"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "llama3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # DeepSeek R1 Distill Qwen 32B NVFP4 — NVFP4 version of the 32B model, Blackwell optimized
  # ═══════════════════════════════════════════════════════════════════════════
  "nm-testing/DeepSeek-R1-Distill-Qwen-32B-NVFP4")
    VLLM_IMAGE="${IMG_EUGR}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=65536
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "deepseek-r1-qwen-32b-nvfp4"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--quantization"        "compressed-tensors"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # DeepSeek R1 Distill Qwen 14B FP8 — 14B reasoning model, FP8, extremely fast on RTX 4080 (12GB)
  # ═══════════════════════════════════════════════════════════════════════════
  "neuralmagic/DeepSeek-R1-Distill-Qwen-14B-FP8")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=65536
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "deepseek-r1-qwen-14b-fp8"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--quantization"        "fp8"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # DeepSeek R1 Distill Qwen 14B AWQ — 14B reasoning model, AWQ, blazing fast on 12GB VRAM
  # ═══════════════════════════════════════════════════════════════════════════
  "casperhansen/deepseek-r1-distill-qwen-14b-awq")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "deepseek-r1-qwen-14b-awq"
      "--dtype"               "float16"
      "--load-format"         "fastsafetensors"
      "--quantization"        "awq"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_xml"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # DeepSeek R1 Distill Llama 8B FP8 — 8B reasoning model, FP8, ultra-fast on RTX 4080 (12GB)
  # ═══════════════════════════════════════════════════════════════════════════
  "neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=65536
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=16

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "deepseek-r1-llama-8b-fp8"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--quantization"        "fp8"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "llama3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # DeepSeek R1 Distill Llama 8B AWQ — 8B reasoning model, AWQ, lightweight and high speed
  # ═══════════════════════════════════════════════════════════════════════════
  "casperhansen/deepseek-r1-distill-llama-8b-awq")
    VLLM_IMAGE="${IMG_NIGHTLY}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=16

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "deepseek-r1-llama-8b-awq"
      "--dtype"               "float16"
      "--load-format"         "fastsafetensors"
      "--quantization"        "awq"
      "--kv-cache-dtype"      "fp8"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "llama3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 on a single GB10
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4")
    VLLM_IMAGE="${IMG_NEMOTRON_OMNI}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=131072
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_NVFP4_GEMM_BACKEND=marlin
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e VLLM_USE_FLASHINFER_MOE_FP4=0
      -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      -e CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      -e OMP_NUM_THREADS=4
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"         "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
      "--served-model-name"         "nemotron-3-nano-omni"
      "--quantization"              "fp4"
      "--moe-backend"               "marlin"
      "--kv-cache-dtype"            "fp8"
      "--mamba-ssm-cache-dtype"     "float32"
      "--reasoning-parser"          "nemotron_v3"
      "--enable-auto-tool-choice"
      "--tool-call-parser"          "qwen3_coder"
      "--video-pruning-rate"        "0.5"
      "--limit-mm-per-prompt"       '{"video":1,"image":1,"audio":1}'
      "--media-io-kwargs"           '{"video":{"fps":2,"num_frames":256}}'
      "--allowed-local-media-path"  "/"
      "--trust-remote-code"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Z.AI GLM-5.1-FP8 754B MoE on 8 DGX Spark nodes
  # ═══════════════════════════════════════════════════════════════════════════
  "zai-org/GLM-5.1-FP8")
    VLLM_IMAGE="${IMG_GLM51}"
    GPU_MEM_UTIL=0.84
    MAX_MODEL_LEN=24576
    MAX_BATCHED_TOKENS=8192
    MAX_NUM_SEQS=8
    MODEL_DOWNLOADS=("none")

    VOLUME_ARGS=(
      -v /mnt/glm51-hf-cache:/root/.cache/huggingface
      -v /mnt/glm51-hf-cache:/mnt/glm51-hf-cache
    )

    ENV_ARGS=(
      -e HF_HUB_OFFLINE=1
      -e TRANSFORMERS_OFFLINE=1
      -e HF_HOME=/mnt/glm51-hf-cache
      -e VLLM_USE_DEEP_GEMM=0
      -e VLLM_MOE_USE_DEEP_GEMM=0
      -e VLLM_USE_DEEP_GEMM_E8M0=0
      -e VLLM_MLA_DISABLE=1
      -e VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel
      -e VLLM_USE_FLASHINFER_SAMPLER=0
      -e OMP_NUM_THREADS=4
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--tensor-parallel-size"        "8"
      "--distributed-executor-backend" "ray"
      "--enforce-eager"
      "--kv-cache-dtype"              "fp8"
      "--tool-call-parser"            "glm47"
      "--reasoning-parser"            "glm45"
      "--enable-auto-tool-choice"
      "--chat-template-content-format" "string"
      "--served-model-name"           "glm-5.1-fp8"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # nvidia/MiniMax-M2.7-NVFP4
  # ═══════════════════════════════════════════════════════════════════════════
  "nvidia/MiniMax-M2.7-NVFP4")
    VLLM_IMAGE="${IMG_MINIMAX_NVFP4}"
    GPU_MEM_UTIL=0.85
    MAX_MODEL_LEN=196608
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_MARLIN_USE_ATOMIC_ADD=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"           "minimax-m2.7-nvfp4"
      "-tp"                           "4"
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
    MAX_MODEL_LEN=128000
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"            "minimax-m2.5-awq"
      "-tp"                            "4"
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
    MAX_MODEL_LEN=128000
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"            "minimax-m2.7-awq"
      "-tp"                            "2"
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
      "--tensor-parallel-size"         "8"
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
    MAX_MODEL_LEN=200000
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
      "-tp"                            "2"
      "-pp"                            "1"
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

  *)
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=32768
    MAX_BATCHED_TOKENS=32768
    MAX_NUM_SEQS=128
    ;;
esac

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

  # If no | separator, target_dir equals repo → cache mode
  if [[ "${target_dir}" == "${repo}" ]]; then
    target_dir=""
  fi

  if [[ -n "${target_dir}" ]]; then
    # ── Local directory download ──
    if [[ ! -f "${target_dir}/config.json" && ! -f "${target_dir}/params.json" ]]; then
      echo "📥 Downloading ${repo} → ${target_dir} ..."
      mkdir -p "${target_dir}"
      docker run --rm \
        --entrypoint python3 \
        -v "${target_dir}:${target_dir}" \
        -e HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
        "${VLLM_IMAGE}" \
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
  else
    # ── HuggingFace cache download ──
    local cache_name="models--$(echo "${repo}" | tr '/' '--')"
    local cache_path="${HOME}/.cache/huggingface/hub/${cache_name}"
    if [[ ! -d "${cache_path}/snapshots" ]]; then
      echo "📥 Caching ${repo} ..."
      docker run --rm \
        --entrypoint python3 \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -e HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
        "${VLLM_IMAGE}" \
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

# ── MOD: DROP CACHES ──────────────────────────────────────────────────────────
# If selected model includes drop-caches mod, drop system caches to reclaim RAM
if [[ "${MODEL}" == "zai-org/GLM-5.1-FP8" || "${MODEL}" == "nvidia/Kimi-K2.6-NVFP4" ]]; then
  echo "🧹 Applying mod: mods/drop-caches..."
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo "   Writing to /proc/sys/vm/drop_caches..."
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  else
    echo "   Note: Could not drop caches (insufficient permissions), proceeding..."
  fi
fi

docker stop "${CONTAINER_NAME}" 2>/dev/null
docker rm   "${CONTAINER_NAME}" 2>/dev/null

docker run -d \
  --name    "${CONTAINER_NAME}" \
  --gpus    all \
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
