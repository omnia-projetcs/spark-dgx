#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# mix-vllm.sh — vLLM launcher for DGX Spark (GB10 / SM121, 128 GB)
# Models ranked by single-node throughput (Spark Arena leaderboard)
#
#  Rank  Model                                         tok/s    Quant     Image
#  ────  ──────────────────────────────────────────   ───────  ───────   ────────────
#  #1    AEON-7/Qwen3.6-35B-heretic-NVFP4 + DFlash    88–117   NVFP4    aeon-7 v1.2
#  #2    openai/gpt-oss-120b (MXFP4)                   ~60     MXFP4    eugr-mxfp4 *
#  #3    nvidia/Nemotron-3-Nano-30B-A3B-NVFP4          ~56     NVFP4    eugr-nightly
#  #4    THUDM/glm-4.7-flash-awq                       ~35     AWQ      eugr-tf5
#  #5    Qwen/Qwen3.6-35B-A3B-FP8                      ~30     FP8      cu130-nightly
#  #6    nvidia/Nemotron-3-Super-120B-A12B-NVFP4       ~22     NVFP4    eugr-nightly
#  #7    RedHatAI/Qwen3.5-122B-A10B-NVFP4              ~17     NVFP4    eugr-nightly  ← best quality
#  ──    google/gemma-3-12b-it                        fast     BF16     vllm-latest
#  ──    bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4   ~20     NVFP4    eugr-nightly
#  ──    rdtand/Qwen3.6-35B-A3B-PrismaQuant            ~40     4.75bit  vllm-latest
#  ──    Intel/Qwen3-Coder-Next-int4-AutoRound         ~30     INT4     vllm-latest
#  ──    LiquidAI/LFM2.5-350M                         fast     BF16     vllm-latest
#
#  * gpt-oss MXFP4 needs image built with --exp-mxfp4 from eugr/spark-vllm-docker
#    ./build-and-copy.sh --exp-mxfp4  → produces local image: vllm-node-mxfp4
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
if [[ -z "${HUGGING_FACE_HUB_TOKEN}" || "${HUGGING_FACE_HUB_TOKEN}" == "hf_..." ]]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║  ⚠️  WARNING — Hugging Face token is NOT configured!               ║"
  echo "╠══════════════════════════════════════════════════════════════════════╣"
  echo "║                                                                    ║"
  echo "║  Most models require a valid Hugging Face token to download.       ║"
  echo "║  Without it, gated models (Llama, Gemma, etc.) will FAIL.          ║"
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
  exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

CONTAINER_NAME="mix-vllm"
PORT=8000

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
MODEL="AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4"
# MODEL="openai/gpt-oss-120b"
# MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
# MODEL="THUDM/glm-4.7-flash-awq"
# MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
# MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
# MODEL="RedHatAI/Qwen3.5-122B-A10B-NVFP4"
# MODEL="google/gemma-3-12b-it"
# MODEL="bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4"
# MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"
# MODEL="Intel/Qwen3-Coder-Next-int4-AutoRound"
# MODEL="LiquidAI/LFM2.5-350M"
# ─────────────────────────────────────────────────────────────────────────────

# ── KNOWN IMAGES ─────────────────────────────────────────────────────────────
IMG_AEON7="ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2"
#         └─ vLLM HEAD cu130/sm_120 + FlashInfer 0.6.8 + 8 SM121 patches
#            NVFP4 CUTLASS + DFlash speculative decoding (Qwen3.6 only)

IMG_EUGR="ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"
#         └─ Standard Spark Arena image — NVFP4/FP8/AWQ, Nemotron, Qwen3.5/3.6

IMG_EUGR_TF5="ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest"
#         └─ Same as above + Transformers 5.0 — required for GLM-4.7

IMG_EUGR_MXFP4="vllm-node-mxfp4"
#         └─ Local image — must be built first:
#            ./build-and-copy.sh --exp-mxfp4  (eugr/spark-vllm-docker)
#            Optimised CUTLASS MXFP4 path for gpt-oss — NOT for other models

IMG_NIGHTLY="vllm/vllm-openai:cu130-nightly"
#         └─ Official nightly with CUDA 13.0 + SM121 kernel support
#            Best for FP8 / BF16 models that don't need NVFP4 patches

IMG_STOCK="vllm/vllm-openai:latest"
#         └─ Standard stable image — BF16/FP8, no NVFP4 SM121 patches
# ─────────────────────────────────────────────────────────────────────────────

# Defaults (overridden per model)
VLLM_IMAGE="${IMG_STOCK}"
EXTRA_ARGS=()
ENV_ARGS=()
VOLUME_ARGS=()

case "${MODEL}" in

  # ═══════════════════════════════════════════════════════════════════════════
  # #1 ★ AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4   88–117 tok/s  ← FASTEST
  # ═══════════════════════════════════════════════════════════════════════════
  # Dedicated image: ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
  # 8 SM121 patches, CUTLASS NVFP4 (not Marlin), DFlash spec-decode.
  # DFlash drafter (z-lab/Qwen3.6-35B-A3B-DFlash, ~870 MB) boosts to 117 tok/s.
  #
  # PRE-REQUISITE — download both models before running:
  #   hf download AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 --local-dir /opt/qwen36/model
  #   hf download z-lab/Qwen3.6-35B-A3B-DFlash        --local-dir /opt/qwen36/dflash
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

    VOLUME_ARGS=(
      -v /opt/qwen36/model:/models/qwen36:ro
      -v /opt/qwen36/dflash:/models/qwen36-dflash:ro
    )

    ENV_ARGS=(
      -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      -e TORCH_MATMUL_PRECISION=high
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      -e NVIDIA_FORWARD_COMPAT=1
      -e VLLM_TEST_FORCE_FP8_MARLIN=1
      -e TORCHINDUCTOR_MAX_AUTOTUNE=0
      -e TRITON_MAX_AUTOTUNE=0
      -e VLLM_USE_V1=1
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "/models/qwen36"        # model path override (local mount)
      "--served-model-name"   "qwen36-fast"
      "--dtype"               "auto"
      "--quantization"        "compressed-tensors"
      "--kv-cache-dtype"      "fp8"
      "--tensor-parallel-size" "1"
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
      "--reasoning-parser"    "qwen3"
      "--speculative-config"  '{"method":"dflash","model":"/models/qwen36-dflash","num_speculative_tokens":15}'
    )
    # Override MODEL for docker run command — local path replaces HF ID
    MODEL="/models/qwen36"
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #2 openai/gpt-oss-120b (MXFP4)   ~58–60 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # MXFP4-specific image required (local build from eugr/spark-vllm-docker):
  #   ./build-and-copy.sh --exp-mxfp4
  # Uses native CUTLASS MXFP4 path — do NOT use with other models.
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
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--reasoning-parser"    "harmony"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #3 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4   ~56 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # Best throughput among ready-to-run models (no pre-download needed).
  # MoE: 30B total / 3.5B active. Reasoning + tool-call capable.
  # ⚠️  Requires mod 'nemotron-nano' (nano_v3_reasoning_parser.py).
  #     Mount the plugin from eugr/spark-vllm-docker/mods/nemotron-nano/:
  #       -v /path/to/nano_v3_reasoning_parser.py:/nano_v3_reasoning_parser.py:ro
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
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
      "--reasoning-parser-plugin" "/nano_v3_reasoning_parser.py"
      "--reasoning-parser"    "nano_v3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #4 THUDM/glm-4.7-flash-awq   ~35 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # Fast AWQ MoE by Tsinghua. Single-node capable (~50 GB AWQ).
  # Requires Transformers 5.0 → use eugr-nightly-tf5 image.
  # ⚠️  fix-glm-4.7-flash-AWQ mod applied in this image by default.
  # ═══════════════════════════════════════════════════════════════════════════
  "THUDM/glm-4.7-flash-awq")
    VLLM_IMAGE="${IMG_EUGR_TF5}"
    GPU_MEM_UTIL=0.80
    MAX_MODEL_LEN=131072
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=8

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "glm4-flash"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--quantization"        "awq"
      "--kv-cache-dtype"      "fp8"
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "glm4_moe"
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
      "--served-model-name"   "qwen3.6-35b"
      "--dtype"               "auto"
      "--load-format"         "fastsafetensors"
      "--kv-cache-dtype"      "fp8"
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # #6 nvidia/Nemotron-3-Super-120B-A12B-NVFP4   ~20–22 tok/s
  # ═══════════════════════════════════════════════════════════════════════════
  # Model ~70 GB + KV cache ~34 GB → fits in 128 GB.
  # Marlin dequant (FP4→BF16) — native FP4 compute not yet on SM121 in vLLM.
  # ⚠️  Requires 'nemotron-super' mod (super_v3_reasoning_parser.py):
  #       -v /path/to/super_v3_reasoning_parser.py:/super_v3_reasoning_parser.py:ro
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
      "--enable-prefix-caching"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
      "--reasoning-parser-plugin" "/super_v3_reasoning_parser.py"
      "--reasoning-parser"    "super_v3"
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
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
      "--reasoning-parser"    "qwen3"
    )
    ;;

  # ═══════════════════════════════════════════════════════════════════════════
  # Gemma 3 12B — BF16, ~24 GB, large context 128k, pythonic tool parser
  # ═══════════════════════════════════════════════════════════════════════════
  "google/gemma-3-12b-it")
    VLLM_IMAGE="${IMG_STOCK}"
    GPU_MEM_UTIL=0.50
    MAX_MODEL_LEN=131072
    MAX_BATCHED_TOKENS=16384
    MAX_NUM_SEQS=16

    ENV_ARGS=(
      -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
      -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
    )

    EXTRA_ARGS=(
      "--served-model-name"   "gemma3-12b"
      "--dtype"               "bfloat16"
      "--load-format"         "fastsafetensors"
      "--attention-backend"   "flashinfer"
      "--enable-prefix-caching"
      "--enable-chunked-prefill"
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "pythonic"
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
      "--tensor-parallel-size" "4"
      "--enable-prefix-caching"
      "--async-scheduling"
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
      "--tool-call-parser"    "qwen3_coder"
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
    )

    EXTRA_ARGS=(
      "--enable-auto-tool-choice"
      "--tool-call-parser"    "qwen3_coder"
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

    ENV_ARGS=(-e VLLM_MARLIN_USE_ATOMIC_ADD=1)

    EXTRA_ARGS=(
      "--language-model-only"
      "--load-format"         "fastsafetensors"
      "--attention-backend"   "flash_attn"
      "--dtype"               "auto"
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

# ─────────────────────────────────────────────────────────────────────────────

echo "🔥 ${MODEL}"
echo "   image  : ${VLLM_IMAGE}"
echo "   ctx    : ${MAX_MODEL_LEN}   mem: ${GPU_MEM_UTIL}   seqs: ${MAX_NUM_SEQS}"

docker stop ${CONTAINER_NAME} 2>/dev/null
docker rm   ${CONTAINER_NAME} 2>/dev/null

docker run -d \
  --name    ${CONTAINER_NAME} \
  --gpus    all \
  --ipc     host \
  --ulimit  memlock=-1 \
  --ulimit  stack=67108864 \
  --shm-size 32g \
  -p ${PORT}:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${VOLUME_ARGS[@]}" \
  --restart unless-stopped \
  --entrypoint vllm \
  "${ENV_ARGS[@]}" \
  ${VLLM_IMAGE} \
  serve ${MODEL} \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len          ${MAX_MODEL_LEN} \
    --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \
    --max-num-seqs           ${MAX_NUM_SEQS} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"

echo "✅ ${CONTAINER_NAME} → http://localhost:${PORT}"
echo "📋 Logs   : docker logs -f ${CONTAINER_NAME}"
echo "🔍 Health : curl http://localhost:${PORT}/health"
