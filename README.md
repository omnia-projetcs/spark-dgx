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
| ★1 | `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` | ~71 (88–117) | NVFP4 | 128K | 💬 🔧 🧠 | DFlash spec-decode, CUTLASS NVFP4, custom aeon-7 image, FP8 KV-cache |
| #2 | `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` | ~59 | 4.75bit | 256K | 💬 🔧 🧠 | Speculative decoding (MTP ×3), FP8 KV-cache |
| #3 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | ~58 | NVFP4 | 256K | 💬 🔧 🧠 | MoE 30B/3.5B active, FlashInfer FP4 |
| #4 | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | ~50 | NVFP4 | 262K | 💬 🖼️ 🎥 🔊 🔧 🧠 | Multimodal, TP×4, FP8 KV-cache |
| #5 | `Qwen/Qwen3.6-35B-A3B-FP8` | ~30 | FP8 | 256K | 💬 🔧 🧠 | 156 tok/s aggregate (c=32), cu130-nightly |
| #6 | `rdtand/MiniMax-M2.7-PrismaQuant-3.20bit-vllm` | ~25 | 3.20bit | 196K | 💬 🔧 🧠 | MiniMax M2.7, PrismaQuant 3.20bit, standard eugr image |
| #7 | `Intel/Qwen3-Coder-Next-int4-AutoRound` | ~17 | INT4 | 1M | 💬 🔧 | MoE FP8, YaRN RoPE scaling, 384 concurrent sequences |
| #8 | `RedHatAI/Qwen3.5-122B-A10B-NVFP4` | ~17 | NVFP4 | 64K | 💬 🔧 🧠 | **Best quality** — RedHat calibration ≈ FP16, FlashInfer |
| — | `shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC` | — | INT4 | 196K | 💬 🔧 | AutoRound INT4, z-lab DFlash speculative decoding, custom vllm-node-tf5 image |
| #9 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | ~15 | NVFP4 | 128K | 💬 🔧 🧠 | MoE 120B/12B active, Marlin dequant |
| — | `LiquidAI/LFM2.5-350M` | ~212 | BF16 | 32K | 💬 | Ultra-lightweight 350M, ideal for testing/development |
| — | `Qwen/Qwen3.5-0.8B` | ~103 | BF16 | 102K | 💬 🔧 | Lightweight 800M dense model, extremely fast, 102k context |
| — | `nvidia/MiniMax-M2.7-NVFP4` | ~24 | NVFP4 | 196K | 💬 🔧 🧠 | MiniMax M2.7, TP×4, instanttensor format, FP8 KV-cache |
| — | `cyankiwi/MiniMax-M2.5-AWQ-4bit` | cluster | AWQ | 128K | 💬 🔧 🧠 | MiniMax M2.5, TP×4, Ray distributed backend |
| — | `cyankiwi/MiniMax-M2.7-AWQ-4bit` | cluster | AWQ | 128K | 💬 🔧 🧠 | MiniMax M2.7, TP×2, Ray distributed backend |
| — | `nvidia/Kimi-K2.6-NVFP4` | cluster | NVFP4 | 32K | 💬 🔧 🧠 | Kimi K2.6 MoE, TP×8 Ray cluster, shared HF cache, drop-caches mod |
| — | `deepseek-ai/DeepSeek-V4-Flash` | cluster | FP8 | 200K | 💬 🔧 🧠 | DeepSeek V4 Flash FP8, TP=2 MP, custom vllm-node-dsv4 image |

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

### 📊 Real-World Benchmark Results (Single-Node)

The following benchmark runs represent verified performance metrics across different concurrency levels (1, 2, 4, and 8 simultaneous streams) on a single **NVIDIA DGX Spark** deskside supercomputer (GB10 Grace Blackwell, 128GB unified memory).

Click on any model below to expand its detailed concurrency comparison table:

<details>
<summary><b>1. AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │    120.6ms │    133.8ms │    7.622s │     512.0 │     70.59 │    457.74
      2 │    397.9ms │   1012.9ms │   10.344s │     512.0 │     53.71 │    325.83
      4 │   8069.4ms │  12001.9ms │   18.381s │     512.0 │     50.33 │    173.43
      8 │  15521.6ms │  31851.4ms │   25.757s │     512.0 │     50.84 │     98.34
```

</details>

<details>
<summary><b>2. rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │    123.0ms │    142.3ms │    8.936s │     512.0 │     58.66 │    401.92
      2 │    513.2ms │   1247.5ms │   10.452s │     512.0 │     52.01 │    338.13
      4 │   1104.7ms │   3806.2ms │   14.220s │     512.0 │     40.09 │    243.52
      8 │   5347.8ms │  10481.9ms │   10.482s │      77.6 │      7.56 │     59.23
```

</details>

<details>
<summary><b>3. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │    111.4ms │    320.9ms │    8.951s │     512.0 │     57.92 │    445.74
      2 │    217.1ms │    340.7ms │   10.050s │     498.6 │     50.72 │    380.15
      4 │    244.4ms │    322.4ms │   13.601s │     493.1 │     36.92 │    277.35
      8 │    206.6ms │    210.2ms │   20.820s │     512.0 │     24.84 │    196.65
```

</details>

<details>
<summary><b>4. bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │     58.0ms │     65.7ms │    9.731s │     479.8 │     49.62 │    367.64
      2 │    262.0ms │    854.7ms │   10.893s │     481.8 │     45.32 │    314.14
      4 │     78.2ms │    114.1ms │   13.586s │     480.8 │     35.64 │    263.44
      8 │    101.5ms │    109.6ms │   17.817s │     478.9 │     27.09 │    200.71
```

</details>

<details>
<summary><b>5. minimax-m2.7-prisma (rdtand/MiniMax-M2.7-PrismaQuant-3.20bit-vllm)</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (minimax-m2.7-prisma)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │    313.6ms │    713.4ms │   21.168s │     512.0 │     24.55 │     24.19
      2 │    297.5ms │    438.7ms │   29.994s │     512.0 │     17.24 │     34.13
      4 │    348.1ms │    384.9ms │   46.086s │     512.0 │     11.19 │     44.41
      8 │   1011.4ms │   1129.8ms │   72.989s │     512.0 │      7.11 │     56.08
```

</details>

<details>
<summary><b>6. Intel/Qwen3-Coder-Next-int4-AutoRound</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (Intel/Qwen3-Coder-Next-int4-AutoRound)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │     39.5ms │     42.8ms │    1.961s │     128.8 │     16.66 │     65.66
      2 │     47.2ms │     57.7ms │    2.924s │     192.6 │     25.03 │     99.55
      4 │     62.0ms │     78.1ms │    1.979s │     129.9 │     61.04 │    133.49
      8 │     99.4ms │    136.4ms │    1.089s │      65.6 │     38.97 │     66.51
```

</details>

<details>
<summary><b>7. nemotron-super-120b (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (nemotron-super-120b)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │    294.5ms │    475.2ms │   35.564s │     512.0 │     14.52 │     14.40
      2 │    585.9ms │    717.3ms │   41.996s │     512.0 │     12.36 │     24.38
      4 │    627.4ms │    699.2ms │   54.939s │     512.0 │      9.43 │     37.27
      8 │  28040.7ms │  55555.8ms │   82.570s │     512.0 │      9.39 │     37.23
```

</details>

<details>
<summary><b>8. LiquidAI/LFM2.5-350M</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (LiquidAI/LFM2.5-350M)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │     34.5ms │    149.5ms │    1.756s │     363.9 │    211.62 │   1169.92
      2 │     54.7ms │    160.8ms │    1.629s │     393.5 │    253.43 │   1284.92
      4 │     21.6ms │     24.2ms │    1.508s │     376.9 │    255.10 │   1358.30
      8 │    749.5ms │   2027.5ms │    2.233s │     379.4 │    255.98 │    934.11
```

</details>

<details>
<summary><b>9. Qwen/Qwen3.5-0.8B</b></summary>

```
══════════════════════════════════════════════════════════════════════
  📊 COMPARISON TABLE (Qwen/Qwen3.5-0.8B)
══════════════════════════════════════════════════════════════════════
   Conc │   TTFT avg │   TTFT p95 │   Lat avg │  Tok/resp │  t/s (req) │  t/s (agg)
  ─────────────────────────────────────────────────────────────────────────────────
      1 │     57.8ms │    263.9ms │    4.782s │     487.4 │    103.23 │    101.92
      2 │     39.7ms │     44.3ms │    4.248s │     477.8 │    113.68 │    208.75
      4 │    110.9ms │    178.0ms │    4.561s │     486.1 │    109.22 │    405.49
      8 │   2247.8ms │   4804.8ms │    6.474s │     462.0 │    109.24 │    392.19
```

</details>

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
*   Run vLLM with Ray or PyTorch Distributed to scale up to **Tensor Parallelism (TP) = 2** or **Pipeline Parallelism (PP) = 2** to run massive models up to 405B