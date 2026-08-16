#!/usr/bin/env bash
set -euo pipefail

# MiniMax H3 Turbo is a Diffusers video model, not a vLLM language model.
# This launcher follows ModelTC's official single-GPU inference workflow.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/minimax-h3-turbo"
REPO_DIR="${MINIMAX_H3_REPO_DIR:-${CACHE_ROOT}/repo}"
CHECKPOINT_DIR="${MINIMAX_H3_CHECKPOINT_DIR:-${CACHE_ROOT}/checkpoints}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/minimax-h3-turbo"
JOBS_JSON=""
VARIANT="4step"
DRY_RUN=false
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./minimax-h3-turbo.sh [options] [-- extra inference arguments]

Options:
  --variant <4step|8step|768p>  Turbo checkpoint (default: 4step)
  --jobs-json <path>           Job file; defaults to the official T2VA examples
  --output-dir <path>          MP4 output directory
  --repo-dir <path>            Existing/target Minimax-H3-Turbo source checkout
  --checkpoint-dir <path>      Directory used for the Turbo LoRA checkpoint
  --dry-run                    Validate jobs without loading model weights
  -h, --help                   Show this help

This model runs on one DGX Spark. It is not served through the OpenAI/vLLM API.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Option $1 requires a value." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      require_value "$@"
      VARIANT="$2"
      shift 2
      ;;
    --jobs-json)
      require_value "$@"
      JOBS_JSON="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      require_value "$@"
      REPO_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      require_value "$@"
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${VARIANT}" in
  4step)
    CHECKPOINT_FILE="minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
    INFERENCE_STEPS=4
    ;;
  8step)
    CHECKPOINT_FILE="minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    INFERENCE_STEPS=8
    ;;
  768p)
    CHECKPOINT_FILE="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    INFERENCE_STEPS=4
    EXTRA_ARGS+=(--video-shift 6 --lora-alpha 128 --megapixels 1.0 --aspect-ratio 16:9)
    ;;
  *)
    echo "Invalid variant '${VARIANT}'; expected 4step, 8step, or 768p." >&2
    exit 2
    ;;
esac

for command_name in git python3 hf; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    if [[ "${command_name}" == "hf" ]]; then
      echo "Install it with: python3 -m pip install -U huggingface_hub" >&2
    fi
    exit 1
  fi
done

if ! python3 -c 'import diffusers, peft, safetensors, torch' >/dev/null 2>&1; then
  echo "Missing MiniMax H3 Python dependencies." >&2
  echo "Install PyTorch, PEFT, safetensors, and a current Diffusers build first:" >&2
  echo "  python3 -m pip install -U torch torchvision peft safetensors transformers accelerate pillow" >&2
  echo "  python3 -m pip install -U git+https://github.com/huggingface/diffusers.git" >&2
  exit 1
fi

if [[ ! -f "${REPO_DIR}/inference_minimax_h3.py" ]]; then
  mkdir -p "$(dirname -- "${REPO_DIR}")"
  echo "Cloning the official MiniMax H3 Turbo inference repository..."
  git clone --depth 1 https://github.com/ModelTC/Minimax-H3-Turbo.git "${REPO_DIR}"
fi

if [[ -z "${JOBS_JSON}" ]]; then
  JOBS_JSON="${REPO_DIR}/examples/prompts_t2va_test.json"
fi

if [[ ! -f "${JOBS_JSON}" ]]; then
  echo "Jobs file not found: ${JOBS_JSON}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_FILE}"

if [[ "${DRY_RUN}" != "true" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Downloading lightx2v/Minimax-h3-Turbo (${CHECKPOINT_FILE})..."
  hf download lightx2v/Minimax-h3-Turbo "${CHECKPOINT_FILE}" --local-dir "${CHECKPOINT_DIR}"
fi

COMMAND=(
  python3 "${REPO_DIR}/inference_minimax_h3.py"
  --jobs-json "${JOBS_JSON}"
  --inference-steps "${INFERENCE_STEPS}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  COMMAND+=(--dry-run)
else
  COMMAND+=(--lora-path "${CHECKPOINT_PATH}")
fi

COMMAND+=("${EXTRA_ARGS[@]}")

echo "Model : lightx2v/Minimax-h3-Turbo (${VARIANT})"
echo "Target: MiniMaxAI/MiniMax-H3"
echo "Output: ${OUTPUT_DIR}"
exec "${COMMAND[@]}"
