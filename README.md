# NVIDIA DGX Spark GB10 — AI Models & Inference Guide

[![NVIDIA Developer Forum](https://img.shields.io/badge/NVIDIA_Developer-Forum-green?logo=nvidia)](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)
[![Spark Arena Leaderboard](https://img.shields.io/badge/Spark_Arena-Leaderboard-blue)](https://spark-arena.com/leaderboard)
[![vLLM Support](https://img.shields.io/badge/vLLM-Supported-orange?logo=python)](https://github.com/vllm-project/vllm)
[![Platform](https://img.shields.io/badge/Platform-Linux_aarch64-lightgrey)](https://www.nvidia.com/en-us/data-center/dgx-spark/)

Welcome to the ultimate repository and guide for running, optimizing, and benchmarking state-of-the-art AI models on the **NVIDIA DGX Spark** deskside supercomputer, powered by the cutting-edge **NVIDIA GB10 Grace Blackwell Superchip**. 

This guide is designed to help researchers, developers, and data scientists get the maximum performance out of their local workstation when running LLMs, VLMs, and diffusion models.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Benchmarks & Leaderboards](#benchmarks--leaderboards)
3. [High-Performance Inference with vLLM](#high-performance-inference-with-vllm)
    - [Why vLLM on DGX Spark?](#why-vllm-on-dgx-spark)
    - [Docker-First Approach: Performance & Cleanliness](#docker-first-approach-performance--cleanliness)
    - [Running vLLM via Docker](#running-vllm-via-docker)
4. [Included Tools](#included-tools)
    - [`mix-vllm.sh` — Multi-Model Launcher](#mix-vllmsh--multi-model-launcher)
    - [`benchmark.py` — Performance Testing](#benchmarkpy--performance-testing)
5. [Useful Resources & External Links](#useful-resources--external-links)
6. [Best Practices for DGX Spark (GB10)](#best-practices-for-dgx-spark-gb10)

---

## System Overview

The **NVIDIA DGX Spark** is a compact, deskside AI supercomputer featuring the revolutionary **NVIDIA GB10 Grace Blackwell Superchip**. It bridges the gap between local prototyping and heavy data center scaling.

### Key Specifications:
*   **Processor:** Coherent ARM64-based NVIDIA Grace CPU + NVIDIA Blackwell GPU on a single chip.
*   **Unified Memory:** 128GB of coherent, unified LPDDR5x system memory shared seamlessly between CPU and GPU via high-speed NVLink-C2C.
*   **Performance:** Up to **1 PetaFLOP** of local AI performance (using FP4 precision).
*   **Networking:** Integrated NVIDIA ConnectX-7 SmartNIC.
*   **Capacity:** Out-of-the-box support for running models up to **200 Billion parameters** locally.
*   **Multi-Node Link:** Two DGX Spark units can be linked to support up to **405 Billion parameter** models.

---

## Benchmarks & Leaderboards

When deploying models locally on your DGX Spark, performance will vary depending on your precision (FP16, FP8, FP4), batch sizes, and tensor parallelism configurations. 

### 🏆 Spark Arena
For real-time and crowd-sourced evaluations of model latency, throughput, token generation speed, and task quality specifically calibrated for such workstations, check the official benchmark index:
👉 **[Spark Arena Leaderboard](https://spark-arena.com/leaderboard)**

To easily launch, manage, and orchestrate LLM inference workloads on one or more NVIDIA DGX Spark systems (without the complexity of Slurm or Kubernetes), check the official management tool:
👉 **[sparkrun Website](https://sparkrun.dev/)**

#### Key Metrics to Track on the Spark Arena:
*   **TTFT (Time to First Token):** Critical for interactive applications (e.g., chatbots).
*   **Inter-Token Latency:** The generation speed (tokens per second).
*   **Throughput (Tokens/sec/GPU):** Important for batched, offline processing workflows.
*   **Quantization Quality Degradation:** Compares quantized variants (like AWQ, GPTQ, FP8) against native FP16 baselines.

### 📈 Performance Experience & Concurrency Reports
For an engineering deep-dive and real-world concurrency benchmark reports on the DGX Spark workstation:
👉 **[Dendro Logic DGX Spark Concurrency Benchmark](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/)**

---

## High-Performance Inference with vLLM

### Why vLLM on DGX Spark?

**vLLM** stands out as one of the absolute **most performant, high-throughput, and memory-efficient engines** to run LLM inference on the NVIDIA DGX Spark platform. By utilizing **PagedAttention**, vLLM dynamically manages the KV-cache, virtually eliminating memory fragmentation and allowing you to maximize the Spark’s 128GB unified Grace Blackwell memory.

### Docker-First Approach: Performance & Cleanliness

To get the absolute best out of vLLM on ARM64 (`aarch64`) / Blackwell, **running vLLM within Docker is strongly recommended and essential**.

> [!IMPORTANT]
> **Why you should avoid bare-metal installation and use Docker:**
> 
> *   **Optimal Performance:** Pre-built ARM64 Docker images are natively compiled with deeply-integrated libraries (optimized PyTorch, specific Triton versions, FlashAttention/FlashInfer backends, and custom CUDA kernels) specifically targeted for Blackwell's computing capability. Manually reproducing these compiling optimizations on the host OS is extremely complex.
> *   **Zero Host Pollution ("Pourrissement de la Machine"):** Local source building installs heavy compilation toolchains, custom Python packages, multiple dependency versions, and complex runtime libraries (`libnuma-dev`, NCCL, specific CUDA Toolkit path modifications) that can clutter the system and break other environments or system-level DGX OS libraries. Docker completely encapsulates the inference stack, keeping the host OS clean, lightweight, and stable.

### Running vLLM via Docker

Use the official pre-optimized ARM64 vLLM containers from NVIDIA NGC, or utilize optimized community-built setups specifically configured for the Grace Blackwell GB10:
*   👉 **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)**: Dedicated Docker setup for running vLLM on DGX Spark.
*   👉 **[saifgithub/vllm-gb10-sm121](https://github.com/saifgithub/vllm-gb10-sm121)**: Optimized vLLM configurations for Grace Blackwell (`sm_121`) target.

Launching the OpenAI-compatible API server takes only a single, sandboxed command:

```bash
docker run --gpus all \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  vllm serve "meta-llama/Meta-Llama-3-8B-Instruct" \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

#### Querying your dockerized vLLM Server:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain the advantages of NVIDIA Grace Blackwell NVLink-C2C connection."}
    ],
    "temperature": 0.7
  }'
```

---

## Included Tools

This repository ships with two ready-to-use scripts to deploy and benchmark models on your DGX Spark.

### `mix-vllm.sh` — Multi-Model Launcher

A turnkey Bash script that launches a fully configured, production-ready **vLLM Docker container** with optimized per-model settings. Simply uncomment the model you want to serve and run the script.

#### Supported Models (pre-configured):

| # | Model | tok/s | Quant | Context | Capabilities | Key Features |
|---|---|---|---|---|---|---|
| ★1 | `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` | 88–117 | NVFP4 | 128K | 💬 🔧 🧠 | DFlash spec-decode, CUTLASS NVFP4, custom aeon-7 image, FP8 KV-cache |
| #2 | `openai/gpt-oss-120b` | ~60 | MXFP4 | 64K | 💬 🧠 | CUTLASS MXFP4 path, FlashInfer, requires local image build |
| #3 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | ~56 | NVFP4 | 256K | 💬 🔧 🧠 | MoE 30B/3.5B active, FlashInfer FP4 |
| #4 | `THUDM/glm-4.7-flash-awq` | ~35 | AWQ | 128K | 💬 🔧 | Transformers 5.0 image, GLM-4 MoE tool parser |
| #5 | `Qwen/Qwen3.6-35B-A3B-FP8` | ~30 | FP8 | 256K | 💬 🔧 🧠 | 156 tok/s aggregate (c=32), cu130-nightly |
| #6 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | ~22 | NVFP4 | 128K | 💬 🔧 🧠 | MoE 120B/12B active, Marlin dequant |
| #7 | `RedHatAI/Qwen3.5-122B-A10B-NVFP4` | ~17 | NVFP4 | 64K | 💬 🔧 🧠 | **Best quality** — RedHat calibration ≈ FP16, FlashInfer |
| — | `google/gemma-3-12b-it` | fast | BF16 | 128K | 💬 🖼️ 🔧 | Multimodal, pythonic tool parser, 24 GB |
| — | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | ~20 | NVFP4 | 262K | 💬 🖼️ 🎥 🔊 🔧 🧠 | Multimodal, TP×4, FP8 KV-cache |
| — | `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` | ~40 | 4.75bit | 256K | 💬 🔧 🧠 | Speculative decoding (MTP ×3), FP8 KV-cache |
| — | `Intel/Qwen3-Coder-Next-int4-AutoRound` | ~30 | INT4 | 1M | 💬 🔧 | MoE FP8, YaRN RoPE scaling, 384 concurrent sequences |
| — | `LiquidAI/LFM2.5-350M` | fast | BF16 | 32K | 💬 | Ultra-lightweight 350M, ideal for testing/development |

> 💬 Text &nbsp; 🖼️ Image &nbsp; 🎥 Video &nbsp; 🔊 Audio &nbsp; 🔧 Tool-call (MCP-compatible) &nbsp; 🧠 Reasoning/thinking

> [!WARNING]
> **Hugging Face Token Required** — Most models need a valid Hugging Face access token to download weights. Without it, gated models (Llama, Gemma, etc.) will **fail to start**.
>
> **How to get your token:**
> 1. Create an account on [huggingface.co/join](https://huggingface.co/join)
> 2. Go to **Settings → Access Tokens**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> 3. Click **"Create new token"**
> 4. Choose a name (e.g. `dgx-spark`) and select permission **"Read"** (or *Fine-grained* with at least *Read access to contents of all repos under your personal namespace*)
> 5. Copy the token (starts with `hf_...`)
>
> **Configure it** (pick one):
> ```bash
> # Option A — Edit the script directly (line ~40)
> HUGGING_FACE_HUB_TOKEN="hf_YourTokenHere"
>
> # Option B — Export as environment variable before running
> export HUGGING_FACE_HUB_TOKEN="hf_YourTokenHere"
> ```
>
> ⚠️ For **gated models** (Llama, Gemma, etc.), you must also **accept the model license** on its [Hugging Face model page](https://huggingface.co/models) before downloading.
>
> 📖 **Documentation**: [Hugging Face — Security Tokens](https://huggingface.co/docs/hub/en/security-tokens)

#### Usage:
```bash
# 1. Edit mix-vllm.sh to uncomment the desired MODEL line
# 2. Launch the container
./mix-vllm.sh

# Check server health
curl http://localhost:8000/health

# View live logs
docker logs -f mix-vllm
```

Each model configuration includes optimized values for `--gpu-memory-utilization`, `--max-model-len`, `--max-num-batched-tokens`, attention backends, quantization settings, and tool-call parsers.

---

### `benchmark.py` — Performance Testing

A standalone Python script that automatically benchmarks the currently running vLLM server across single-thread and multi-thread concurrency levels.

#### Metrics Reported:

| Metric | Description |
|---|---|
| **TTFT (ms)** | Time To First Token — initial response latency |
| **Tokens/s (per request)** | Generation speed for individual requests |
| **Tokens/s (aggregate)** | Total throughput across all concurrent requests |
| **Tokens/response** | Average number of output tokens per response |
| **Avg latency (s)** | Average total response time per request |

#### Features:
*   **Auto-detects** the served model name from the `/v1/models` endpoint
*   **Streaming SSE** parsing for precise TTFT measurement
*   **Handles `reasoning_content`** from Qwen3 thinking mode
*   **Warmup phase** to avoid cold-start skew
*   **Comparison table** across all concurrency levels
*   **JSON export** of results to `benchmark_results.json`
*   **Zero heavy dependencies** — only requires `requests`

#### Usage:
```bash
# Install dependency
pip install requests

# Default: auto-detect model, test concurrency 1/2/4/8, 8 requests each
python3 benchmark.py

# Custom concurrency levels and more requests
python3 benchmark.py --concurrency 1 4 16 --num-requests 16

# Longer outputs for throughput testing
python3 benchmark.py --max-tokens 1024

# Use a custom prompt file (one prompt per line)
python3 benchmark.py --prompt-file my_prompts.txt

# Target a remote DGX Spark
python3 benchmark.py --base-url http://192.168.1.100:8000
```

#### Example Output:
```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (nemotron-nano-30b)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ──────────────────────────────────────────────────────────────────────
      1 │    120.3ms │    145.2ms │    4.521s │     312.4 │      69.12 │      69.12
      4 │    189.7ms │    234.1ms │    7.832s │     298.1 │      38.07 │     152.28
      8 │    312.5ms │    489.3ms │   12.145s │     285.7 │      23.52 │     188.16
```

---

## Useful Resources & External Links

To stay updated, ask questions, and troubleshoot bugs, utilize the following community hubs and official pages:

### 💬 Forums & Support
*   **[NVIDIA DGX Spark GB10 Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)**: The official channel to report hardware, kernel driver, or system stability bugs.
*   **[r/LocalLLaMA Reddit Community](https://www.reddit.com/r/LocalLLaMA/)**: The largest community for local LLM deployment, hardware setups, configurations, and benchmarks.
*   **[spark-vllm-docker (eugr)](https://github.com/eugr/spark-vllm-docker)**: Highly-targeted, community-maintained Docker deployment resource for vLLM on DGX Spark.
*   **[vllm-gb10-sm121 (saifgithub)](https://github.com/saifgithub/vllm-gb10-sm121)**: Community-maintained build recipes and optimized configurations specifically targeted for GB10 `sm_121`.
*   **[vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)**: Best for library bugs, Triton errors, or unsupported model operators.

### 📚 Official Documentation
*   **[sparkrun Workload Orchestrator](https://sparkrun.dev/)**: Launch, manage, and stop LLM inference workloads on one or more NVIDIA DGX Spark systems — no Slurm, no Kubernetes, no fuss.
*   **[vLLM Official Documentation](https://docs.vllm.ai/)**: For advanced configurations, speculative decoding, and pipeline parallelism.
*   **[NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)**: Official page highlighting Blackwell's internal tech (FP4 Decompression, Dequantization Engine).
*   **[NVIDIA NGC Container Catalog](https://catalog.ngc.nvidia.com/)**: For finding official GPU-optimized, ARM64 vLLM and TensorRT-LLM container tags.

### ⚙️ Alternative Inference & Serving Backends
If you want to test engines other than vLLM:
*   **[Atlas](https://github.com/Avarok-Cybersecurity/atlas)**: An extremely fast and lightweight inference and deployment framework optimized for NVIDIA DGX systems.
*   **[NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)**: NVIDIA's proprietary, hyper-optimized engine. Excellent support for Blackwell FP4/FP8 quantization.
*   **[Ollama (ARM64 Native)](https://github.com/ollama/ollama)**: Super easy to install for simple terminal-based interaction.
*   **[SGLang](https://github.com/sgl-project/sglang)**: Extremely high-throughput server, alternative to vLLM.

---

## Best Practices for DGX Spark (GB10)

### 1. Leverage Coherent Unified Memory
Because the Grace CPU and Blackwell GPU share 128GB of LPDDR5x memory over a high-speed coherent link (up to ~300 GB/s bidirectional, with a total system memory bandwidth of ~273 GB/s), CPU-offloading penalties are significantly lower than standard PCIe-based setups. 
*   If your model is slightly too large for the GPU's immediate workspace, don't hesitate to utilize CPU-offloading strategies or KV-cache offloading configurations inside vLLM (`--gpu-memory-utilization` adjustments).

### 2. Go Quantized (FP8 & FP4)
The Blackwell architecture has specialized hardware support for **FP4** and **FP8** numeric formats, maintaining high precision while doubling throughput.
*   Use models optimized for FP8 (e.g., AWQ/GPTQ or native FP8 checkpoints).
*   This will allow you to fit models up to **70B parameters** or more comfortably with extremely high speeds on a single DGX Spark node.

### 3. Multi-Node Scaling
Using two DGX Spark units via their ConnectX-7 high-speed interfaces enables seamless model partitioning.
*   Run vLLM with Ray or PyTorch Distributed to scale up to **Tensor Parallelism (TP) = 2** or **Pipeline Parallelism (PP) = 2** to run massive models up to 405B parameters.

---
*Contributions to this guide are welcome! If you find any optimized compilation flags or pre-built ARM64 Docker configurations for new models, please open a PR.*