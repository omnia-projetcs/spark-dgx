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
4. [Useful Resources & External Links](#useful-resources--external-links)
5. [Best Practices for DGX Spark (GB10)](#best-practices-for-dgx-spark-gb10)

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

## Useful Resources & External Links

To stay updated, ask questions, and troubleshoot bugs, utilize the following community hubs and official pages:

### 💬 Forums & Support
*   **[NVIDIA DGX Spark GB10 Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)**: The official channel to report hardware, kernel driver, or system stability bugs.
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