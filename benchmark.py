#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  DGX Spark — vLLM Benchmark Script                                 ║
║  Autonomous single-thread & multi-thread performance test           ║
║  Compatible with mix-vllm.sh OpenAI-compatible server               ║
╚══════════════════════════════════════════════════════════════════════╝

Metrics reported:
  • TTFT         — Time To First Token (initial response latency)
  • Tokens/s     — Generation throughput (output tokens per second)
  • Tokens/resp  — Number of output tokens per response
  • Avg latency  — Average total response time per request

Usage:
  python3 benchmark.py                        # auto-detect model on localhost:8000
  python3 benchmark.py --base-url http://10.0.0.5:8000
  python3 benchmark.py --concurrency 1 2 4 8 16
  python3 benchmark.py --prompt-file prompts.txt

Requirements:
  pip install requests  (standard library otherwise)
"""

import argparse
import json
import sys
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import requests
except ImportError:
    print("❌ Missing dependency: pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RequestResult:
    """Result from a single chat completion request."""
    ttft: float             # Time to first token (seconds)
    total_time: float       # Total request time (seconds)
    output_tokens: int      # Number of output tokens generated
    tokens_per_sec: float   # Output tokens / total generation time
    success: bool = True
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated result from a benchmark run."""
    concurrency: int
    num_requests: int
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> List[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    def summary(self) -> dict:
        ok = self.successful
        if not ok:
            return {"error": "All requests failed"}
        ttfts = [r.ttft for r in ok]
        latencies = [r.total_time for r in ok]
        tps_list = [r.tokens_per_sec for r in ok]
        token_counts = [r.output_tokens for r in ok]
        total_tokens = sum(token_counts)
        wall_time = max(latencies)  # approximate wall-clock for concurrent
        return {
            "concurrency": self.concurrency,
            "requests_total": self.num_requests,
            "requests_ok": len(ok),
            "requests_failed": self.failed_count,
            "ttft_avg_ms": round(statistics.mean(ttfts) * 1000, 1),
            "ttft_median_ms": round(statistics.median(ttfts) * 1000, 1),
            "ttft_p95_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)] * 1000, 1),
            "latency_avg_s": round(statistics.mean(latencies), 3),
            "latency_median_s": round(statistics.median(latencies), 3),
            "tokens_per_resp_avg": round(statistics.mean(token_counts), 1),
            "tokens_per_resp_total": total_tokens,
            "tps_per_request_avg": round(statistics.mean(tps_list), 2),
            "tps_aggregate": round(total_tokens / wall_time, 2) if wall_time > 0 else 0,
        }


# ═══════════════════════════════════════════════════════════════════════
# Benchmark prompts — diverse tasks to stress different capabilities
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms, with a real-world analogy.",
    "Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization. Include docstrings and type hints.",
    "Compare and contrast the economic policies of Keynesian economics and supply-side economics. Provide concrete historical examples.",
    "Describe the step-by-step process of how mRNA vaccines work, from injection to immune response.",
    "Write a short science fiction story (about 200 words) set on a space station orbiting Jupiter.",
    "What are the main differences between ARM and x86 CPU architectures? Explain the trade-offs for AI workloads.",
    "Explain how transformers and the attention mechanism work in large language models. Use analogies where possible.",
    "List and explain 5 best practices for securing a Docker container in production environments.",
]


# ═══════════════════════════════════════════════════════════════════════
# Core benchmark logic
# ═══════════════════════════════════════════════════════════════════════

def detect_model(base_url: str) -> str:
    """Auto-detect the served model name from the vLLM /v1/models endpoint."""
    url = f"{base_url}/v1/models"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if not models:
            print("❌ No models found on the server.")
            sys.exit(1)
        model_id = models[0]["id"]
        return model_id
    except Exception as e:
        print(f"❌ Cannot reach vLLM server at {url}: {e}")
        sys.exit(1)


def check_server_health(base_url: str) -> bool:
    """Check if the vLLM server is healthy and ready."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def run_single_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> RequestResult:
    """
    Send a single streaming chat completion request and measure:
    - TTFT (time from request sent to first token received)
    - Total time
    - Output token count
    - Tokens/second
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_token = None
    output_tokens = 0
    full_response = ""

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                # Also count reasoning_content if present (Qwen3 thinking mode)
                reasoning = delta.get("reasoning_content", "")

                token_text = content or reasoning
                if token_text:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    output_tokens += 1
                    full_response += token_text

        t_end = time.perf_counter()

        if t_first_token is None:
            t_first_token = t_end

        total_time = t_end - t_start
        ttft = t_first_token - t_start
        generation_time = t_end - t_first_token
        tps = output_tokens / generation_time if generation_time > 0 else 0.0

        return RequestResult(
            ttft=ttft,
            total_time=total_time,
            output_tokens=output_tokens,
            tokens_per_sec=tps,
        )

    except Exception as e:
        t_end = time.perf_counter()
        return RequestResult(
            ttft=0,
            total_time=t_end - t_start,
            output_tokens=0,
            tokens_per_sec=0,
            success=False,
            error=str(e),
        )


def run_benchmark(
    base_url: str,
    model: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_requests: int = 0,
) -> BenchmarkResult:
    """
    Run a benchmark with the given concurrency level.
    - concurrency=1 → sequential (single-thread)
    - concurrency>1 → parallel (multi-thread)
    """
    if num_requests <= 0:
        num_requests = max(len(prompts), concurrency * 2)

    # Cycle through prompts to fill the request count
    request_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    result = BenchmarkResult(concurrency=concurrency, num_requests=num_requests)

    if concurrency == 1:
        # Sequential execution
        for i, prompt in enumerate(request_prompts):
            r = run_single_request(base_url, model, prompt, max_tokens, temperature)
            result.results.append(r)
            status = "✅" if r.success else "❌"
            print(
                f"  {status} [{i+1}/{num_requests}] "
                f"TTFT={r.ttft*1000:.0f}ms | "
                f"{r.output_tokens} tok | "
                f"{r.tokens_per_sec:.1f} t/s | "
                f"total={r.total_time:.2f}s"
            )
    else:
        # Concurrent execution
        lock = threading.Lock()
        completed = [0]

        def task(idx, prompt):
            r = run_single_request(base_url, model, prompt, max_tokens, temperature)
            with lock:
                completed[0] += 1
                status = "✅" if r.success else "❌"
                print(
                    f"  {status} [{completed[0]}/{num_requests}] "
                    f"TTFT={r.ttft*1000:.0f}ms | "
                    f"{r.output_tokens} tok | "
                    f"{r.tokens_per_sec:.1f} t/s | "
                    f"total={r.total_time:.2f}s"
                )
            return r

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(task, i, p): i
                for i, p in enumerate(request_prompts)
            }
            for future in as_completed(futures):
                result.results.append(future.result())

    return result


# ═══════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════

def print_header(title: str):
    width = 70
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def print_summary(summary: dict):
    if "error" in summary:
        print(f"  ❌ {summary['error']}")
        return
    print(f"  Concurrency       : {summary['concurrency']}")
    print(f"  Requests          : {summary['requests_ok']}/{summary['requests_total']} OK"
          f" ({summary['requests_failed']} failed)")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  TTFT avg          : {summary['ttft_avg_ms']:.1f} ms")
    print(f"  TTFT median       : {summary['ttft_median_ms']:.1f} ms")
    print(f"  TTFT p95          : {summary['ttft_p95_ms']:.1f} ms")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Latency avg       : {summary['latency_avg_s']:.3f} s")
    print(f"  Latency median    : {summary['latency_median_s']:.3f} s")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Tokens/response   : {summary['tokens_per_resp_avg']:.1f} avg  "
          f"({summary['tokens_per_resp_total']} total)")
    print(f"  Tokens/s (per req): {summary['tps_per_request_avg']:.2f} t/s")
    print(f"  Tokens/s (agg.)   : {summary['tps_aggregate']:.2f} t/s  ← total throughput")


def print_comparison_table(all_summaries: List[dict]):
    """Print a compact comparison table across concurrency levels."""
    print_header("📊 COMPARISON TABLE")
    header = (
        f"  {'Conc':>5} │ {'TTFT avg':>10} │ {'TTFT p95':>10} │ "
        f"{'Lat avg':>9} │ {'Tok/resp':>9} │ {'t/s (req)':>10} │ {'t/s (agg)':>10}"
    )
    separator = "  " + "─" * (len(header) - 2)
    print(header)
    print(separator)
    for s in all_summaries:
        if "error" in s:
            continue
        print(
            f"  {s['concurrency']:>5} │ "
            f"{s['ttft_avg_ms']:>8.1f}ms │ "
            f"{s['ttft_p95_ms']:>8.1f}ms │ "
            f"{s['latency_avg_s']:>8.3f}s │ "
            f"{s['tokens_per_resp_avg']:>9.1f} │ "
            f"{s['tps_per_request_avg']:>9.2f} │ "
            f"{s['tps_aggregate']:>9.2f}"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DGX Spark — vLLM Benchmark (single & multi-thread)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (default: auto-detect from server)",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Concurrency levels to test (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=8,
        help="Number of requests per concurrency level (default: 8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max output tokens per request (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional text file with one prompt per line",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup requests before benchmark (default: 1)",
    )

    args = parser.parse_args()

    # ── Server health check ──
    print_header("🔍 SERVER CHECK")
    if not check_server_health(args.base_url):
        print(f"  ❌ Server at {args.base_url} is not healthy or unreachable.")
        print(f"     Make sure mix-vllm.sh is running and the server is ready.")
        sys.exit(1)
    print(f"  ✅ Server is healthy at {args.base_url}")

    # ── Model detection ──
    model = args.model or detect_model(args.base_url)
    print(f"  🤖 Model: {model}")

    # ── Load prompts ──
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"  📝 Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"  📝 Using {len(prompts)} built-in benchmark prompts")

    # ── Warmup ──
    if args.warmup > 0:
        print_header("🔥 WARMUP")
        for i in range(args.warmup):
            r = run_single_request(
                args.base_url, model, prompts[0],
                max_tokens=64, temperature=0.7,
            )
            status = "✅" if r.success else "❌"
            print(f"  {status} Warmup {i+1}/{args.warmup} — "
                  f"TTFT={r.ttft*1000:.0f}ms, {r.output_tokens} tok")

    # ── Run benchmarks ──
    all_summaries = []

    for conc in sorted(args.concurrency):
        label = "SINGLE-THREAD" if conc == 1 else f"MULTI-THREAD (×{conc})"
        print_header(f"🚀 BENCHMARK — {label}")
        print(f"  Concurrency: {conc} | Requests: {args.num_requests} | "
              f"Max tokens: {args.max_tokens}")
        print()

        bench = run_benchmark(
            base_url=args.base_url,
            model=model,
            prompts=prompts,
            concurrency=conc,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_requests=args.num_requests,
        )

        s = bench.summary()
        all_summaries.append(s)

        print()
        print_summary(s)

    # ── Final comparison ──
    if len(all_summaries) > 1:
        print_comparison_table(all_summaries)

    # ── Save JSON results ──
    results_file = "benchmark_results.json"
    output = {
        "server": args.base_url,
        "model": model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "num_requests_per_level": args.num_requests,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": all_summaries,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  💾 Results saved to {results_file}")
    print()


if __name__ == "__main__":
    main()
