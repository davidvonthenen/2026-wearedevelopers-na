#!/usr/bin/env python3
"""Demonstrate cross-instance KV-cache reuse with two vLLM servers and LMCache MP.

The script performs three measured requests:
1. Cold request to instance B after clearing LMCache.
2. Seed request to instance A using a long shared prefix.
3. Reuse request to instance B with the same long prefix and a different suffix.

It reports TTFT, total request time, and LMCache token-level lookup metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import httpx
from transformers import AutoTokenizer


@dataclass(frozen=True)
class RequestResult:
    phase: str
    endpoint: str
    prompt_tokens: int
    ttft_seconds: float
    total_seconds: float
    output_text: str


METRIC_FRAGMENTS = (
    "lookup_requested",
    "lookup_hit",
    "l1_read",
    "l1_write",
    "num_chunks_loaded",
    "l0_l1_load_throughput",
    "l0_l1_store_throughput",
)


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def build_shared_prefix(tokenizer, target_tokens: int) -> tuple[str, int]:
    if target_tokens < 256:
        raise ValueError("target_tokens must be at least 256")

    paragraph = (
        "KV-cache reuse avoids recomputing attention state for token prefixes that "
        "the model has already processed. In production, repeated prefixes often "
        "contain system instructions, tool schemas, retrieved evidence, policy text, "
        "and conversation history. The serving system must preserve exact token order, "
        "model identity, tokenizer behavior, and isolation boundaries before cached "
        "key-value tensors can be reused safely. This paragraph is synthetic benchmark "
        "content and carries no external facts.\n"
    )

    parts: list[str] = []
    token_ids: list[int] = []
    while len(token_ids) < target_tokens:
        parts.append(paragraph)
        token_ids = tokenizer.encode("".join(parts), add_special_tokens=False)

    prefix = tokenizer.decode(token_ids[:target_tokens], skip_special_tokens=False)
    prefix += "\n\n### End shared context\n### Query\n"
    actual_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    return prefix, actual_tokens


def count_prompt_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def run_completion(
    client: httpx.Client,
    endpoint: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    phase: str,
    max_tokens: int,
) -> RequestResult:
    url = f"{normalize_url(endpoint)}/v1/completions"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []

    with client.stream("POST", url, json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_parts.append(text)

    finished = time.perf_counter()
    if first_token_at is None:
        first_token_at = finished

    return RequestResult(
        phase=phase,
        endpoint=normalize_url(endpoint),
        prompt_tokens=prompt_tokens,
        ttft_seconds=first_token_at - started,
        total_seconds=finished - started,
        output_text="".join(output_parts).strip(),
    )


def post_control(client: httpx.Client, lmcache_url: str, path: str) -> None:
    response = client.post(f"{normalize_url(lmcache_url)}{path}")
    response.raise_for_status()


def read_metrics(client: httpx.Client, lmcache_url: str) -> Dict[str, float]:
    response = client.get(f"{normalize_url(lmcache_url)}/metrics")
    response.raise_for_status()

    values: Dict[str, float] = {}
    for raw_line in response.text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        series = fields[0]
        name = series.split("{", 1)[0]
        if not any(fragment in name for fragment in METRIC_FRAGMENTS):
            continue
        try:
            value = float(fields[-1])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def metric_value(metrics: Dict[str, float], fragment: str) -> float:
    return sum(value for name, value in metrics.items() if fragment in name)


def metric_delta(
    before: Dict[str, float], after: Dict[str, float], fragment: str
) -> float:
    return metric_value(after, fragment) - metric_value(before, fragment)


def wait_for_metric_stable(
    client: httpx.Client,
    lmcache_url: str,
    before: Dict[str, float],
    fragment: str,
    timeout_seconds: float,
    quiet_seconds: float = 0.75,
) -> Dict[str, float]:
    """Wait until a metric has increased and then stopped changing briefly."""
    baseline = metric_value(before, fragment)
    deadline = time.monotonic() + timeout_seconds
    last_value = baseline
    last_change = time.monotonic()
    latest = before

    while time.monotonic() < deadline:
        latest = read_metrics(client, lmcache_url)
        current = metric_value(latest, fragment)
        now = time.monotonic()
        if current != last_value:
            last_value = current
            last_change = now
        if current > baseline and now - last_change >= quiet_seconds:
            return latest
        time.sleep(0.20)
    return latest


def print_result_table(results: Iterable[RequestResult]) -> None:
    rows = list(results)
    print("\nMeasured requests")
    print("=" * 96)
    print(
        f"{'Phase':<24} {'Endpoint':<24} {'Prompt tokens':>13} "
        f"{'TTFT (s)':>10} {'Total (s)':>10}"
    )
    print("-" * 96)
    for result in rows:
        print(
            f"{result.phase:<24} {result.endpoint:<24} "
            f"{result.prompt_tokens:>13d} {result.ttft_seconds:>10.3f} "
            f"{result.total_seconds:>10.3f}"
        )
    print("=" * 96)


def print_metric_deltas(
    title: str, before: Dict[str, float], after: Dict[str, float]
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for fragment in (
        "lookup_requested",
        "lookup_hit",
        "l1_read",
        "l1_write",
        "num_chunks_loaded",
    ):
        print(f"{fragment:<24} {metric_delta(before, after, fragment):>12.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate cross-instance LMCache reuse between two vLLM servers."
    )
    parser.add_argument("--instance-a", default="http://127.0.0.1:8000")
    parser.add_argument("--instance-b", default="http://127.0.0.1:8001")
    parser.add_argument("--lmcache-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--tokenizer",
        default="/home/ubuntu/models/Qwen3-8B",
        help="Local tokenizer/model directory on the Lambda instance.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=12288)
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--store-timeout", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )

    shared_prefix, shared_prefix_tokens = build_shared_prefix(
        tokenizer, args.prefix_tokens
    )
    warm_prefix, _ = build_shared_prefix(tokenizer, args.warmup_tokens)

    seed_prompt = (
        shared_prefix
        + "State the main infrastructure purpose of KV-cache reuse in one sentence."
    )
    probe_prompt = shared_prefix + "Respond with the single word READY."
    warm_prompt = warm_prefix + "Respond with the single word WARM."

    seed_tokens = count_prompt_tokens(tokenizer, seed_prompt)
    probe_tokens = count_prompt_tokens(tokenizer, probe_prompt)
    warm_tokens = count_prompt_tokens(tokenizer, warm_prompt)

    print(f"Tokenizer directory: {args.tokenizer}")
    print(f"Shared-prefix tokens: {shared_prefix_tokens}")
    print(f"Probe prompt tokens: {probe_tokens}")

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        # Warm both model processes before any measured request. This removes model-load,
        # CUDA-graph, and first-request effects from the comparison.
        run_completion(
            client,
            args.instance_a,
            args.model,
            warm_prompt,
            warm_tokens,
            "warm A",
            1,
        )
        run_completion(
            client,
            args.instance_b,
            args.model,
            warm_prompt + " B",
            count_prompt_tokens(tokenizer, warm_prompt + " B"),
            "warm B",
            1,
        )
        time.sleep(1.50)

        # Cold B: LMCache is empty. The request may populate LMCache, so it is cleared
        # again before the seed-and-reuse phase.
        post_control(client, args.lmcache_url, "/cache/clear")
        post_control(client, args.lmcache_url, "/metrics/reset")
        cold_before = read_metrics(client, args.lmcache_url)
        cold_b = run_completion(
            client,
            args.instance_b,
            args.model,
            probe_prompt,
            probe_tokens,
            "cold B",
            args.max_output_tokens,
        )
        cold_after = wait_for_metric_stable(
            client,
            args.lmcache_url,
            cold_before,
            "l1_write",
            args.store_timeout,
        )

        # Seed A, then issue the same measured probe to B. Native vLLM prefix caching
        # should be disabled in the server commands, so a B-side hit comes from LMCache.
        post_control(client, args.lmcache_url, "/cache/clear")
        post_control(client, args.lmcache_url, "/metrics/reset")
        seed_before = read_metrics(client, args.lmcache_url)
        seed_a = run_completion(
            client,
            args.instance_a,
            args.model,
            seed_prompt,
            seed_tokens,
            "seed A",
            args.max_output_tokens,
        )
        after_seed = wait_for_metric_stable(
            client,
            args.lmcache_url,
            seed_before,
            "l1_write",
            args.store_timeout,
        )
        hit_before = after_seed
        shared_hit_b = run_completion(
            client,
            args.instance_b,
            args.model,
            probe_prompt,
            probe_tokens,
            "shared hit B",
            args.max_output_tokens,
        )
        hit_after = wait_for_metric_stable(
            client,
            args.lmcache_url,
            hit_before,
            "lookup_requested",
            args.store_timeout,
            quiet_seconds=0.50,
        )

    print_result_table((cold_b, seed_a, shared_hit_b))
    print_metric_deltas("Cold B LMCache activity", cold_before, cold_after)
    print_metric_deltas("Instance A seed activity", seed_before, after_seed)
    print_metric_deltas("Instance B cross-instance reuse", hit_before, hit_after)

    requested = metric_delta(hit_before, hit_after, "lookup_requested")
    hit = metric_delta(hit_before, hit_after, "lookup_hit")
    hit_rate = hit / requested if requested > 0 else 0.0
    speedup = (
        cold_b.ttft_seconds / shared_hit_b.ttft_seconds
        if shared_hit_b.ttft_seconds > 0
        else float("inf")
    )

    print("\nHeadline")
    print("--------")
    print(f"LMCache token hit rate on B: {hit_rate:.1%} ({hit:.0f}/{requested:.0f})")
    print(f"TTFT ratio, cold B / shared-hit B: {speedup:.2f}x")
    print(f"Cold B output: {cold_b.output_text!r}")
    print(f"Shared-hit B output: {shared_hit_b.output_text!r}")

    if args.output_json:
        payload = {
            "configuration": {
                "instance_a": args.instance_a,
                "instance_b": args.instance_b,
                "lmcache_url": args.lmcache_url,
                "model": args.model,
                "tokenizer": args.tokenizer,
                "target_prefix_tokens": args.prefix_tokens,
                "actual_shared_prefix_tokens": shared_prefix_tokens,
            },
            "results": [asdict(result) for result in (cold_b, seed_a, shared_hit_b)],
            "lmcache": {
                "reuse_lookup_requested": requested,
                "reuse_lookup_hit": hit,
                "reuse_hit_rate": hit_rate,
            },
            "ttft_speedup_ratio": speedup,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")

    if requested <= 0:
        print(
            "ERROR: LMCache did not report a completed token lookup. Check connector "
            "configuration, server logs, and the metrics endpoint.",
            file=sys.stderr,
        )
        return 2
    if hit <= 0:
        print(
            "ERROR: Instance B reported no LMCache hit tokens. Confirm that both vLLM "
            "processes point to the same LMCache server and use the same model/tokenizer.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
