"""TurboQuant Auto Benchmark — Iterative testing with progressively larger models.

Runs in background, logs progress to turboquant/auto_benchmark.log.
Tests each available model with various K/V bit configurations to find optimal settings.

Run: python turboquant/auto_benchmark.py
"""

import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

LOG_PATH = Path(__file__).parent / "auto_benchmark.log"
RESULTS_PATH = Path(__file__).parent / "AUTO_BENCHMARK_RESULTS.md"

# Test text — diverse, non-repeated, ~300 tokens
TEST_TEXT = """In 1905, Albert Einstein published four groundbreaking papers that fundamentally changed physics. The photoelectric effect paper proposed that light consists of discrete quanta, which earned him the Nobel Prize in 1921. His paper on Brownian motion provided empirical evidence for the existence of atoms. The special relativity paper introduced the concept that the laws of physics are the same for all non-accelerating observers. The development of quantum mechanics in the 1920s was equally revolutionary. Werner Heisenberg formulated matrix mechanics in 1925, while Erwin Schrodinger developed wave mechanics in 1926. Paul Dirac showed that both formulations were mathematically equivalent. The Copenhagen interpretation proposed that quantum systems exist in superpositions of states until measured. General relativity predicts that massive objects curve spacetime. The Standard Model describes three of the four fundamental forces and classifies elementary particles into quarks, leptons, and gauge bosons. The discovery of the Higgs boson at CERN in 2012 confirmed the mechanism by which particles acquire mass."""

# Models to test, in order of size
MODELS = [
    "gpt2",                        # 124M, 12L, head_dim=64
    "gpt2-medium",                 # 355M, 24L, head_dim=64
    "gpt2-large",                  # 774M, 36L, head_dim=64
    "Qwen/Qwen2.5-0.5B",          # 0.5B, 24L, head_dim=64
    "Qwen/Qwen2.5-1.5B",          # 1.5B, 28L, head_dim=128
    "microsoft/phi-2",             # 2.7B, 32L, head_dim=80
    "Qwen/Qwen2.5-3B",            # 3B,  36L, head_dim=128
    "Qwen/Qwen2.5-7B",            # 7B,  28L, head_dim=128
]

# Bit configurations to test (K_bits, V_bits)
BIT_CONFIGS = [
    (8, 8),   # Reference: 8-bit uniform
    (8, 4),   # Asymmetric: high K, medium V
    (8, 3),   # Asymmetric: high K, low V
    (6, 4),   # Moderate
    (6, 3),   # Moderate
    (5, 3),   # Aggressive
    (4, 4),   # Symmetric low
    (4, 3),   # Very aggressive
    (3, 3),   # Extreme
]


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def eval_ppl(model, input_ids, cache=None) -> float:
    with torch.no_grad():
        if cache is not None:
            out = model(input_ids, past_key_values=cache, use_cache=True)
        else:
            out = model(input_ids, use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    loss = torch.nn.CrossEntropyLoss()(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )
    return math.exp(loss.item())


def get_model_info(model_name: str) -> dict | None:
    """Get model config without downloading weights."""
    try:
        config = AutoConfig.from_pretrained(model_name)
        head_dim = config.hidden_size // config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        params = None  # Can't easily get without loading
        return {
            "name": model_name,
            "layers": config.num_hidden_layers,
            "heads": config.num_attention_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "hidden": config.hidden_size,
        }
    except Exception as e:
        log(f"  Cannot access {model_name}: {e}")
        return None


def test_model(model_name: str, results: list):
    """Test a single model with all bit configurations."""
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from turboquant.kv_cache import make_turboquant_cache

    log(f"\n{'='*60}")
    log(f"Testing: {model_name}")
    log(f"{'='*60}")

    info = get_model_info(model_name)
    if info is None:
        return

    log(f"  Config: {info['layers']}L, {info['heads']}H ({info['kv_heads']}KV), "
        f"head_dim={info['head_dim']}, hidden={info['hidden']}")

    # Check if head_dim >= 3 (minimum for TurboQuant)
    if info["head_dim"] < 3:
        log(f"  SKIP: head_dim={info['head_dim']} too small")
        return

    # Load model
    try:
        log(f"  Loading model...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        model.eval()
        load_time = time.time() - t0
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        log(f"  Loaded: {params_m:.0f}M params in {load_time:.1f}s")
    except Exception as e:
        log(f"  FAIL to load: {e}")
        return

    # Tokenize
    tokens = tokenizer.encode(TEST_TEXT)
    max_pos = getattr(model.config, "max_position_embeddings", 1024)
    if len(tokens) > max_pos:
        tokens = tokens[:max_pos]
    input_ids = torch.tensor([tokens])
    log(f"  Tokens: {len(tokens)}")

    # Baseline
    try:
        log(f"  Computing baseline...")
        ppl_base = eval_ppl(model, input_ids)
        log(f"  Baseline PPL: {ppl_base:.2f}")
    except Exception as e:
        log(f"  FAIL baseline: {e}")
        del model; gc.collect()
        return

    # Get KV norms for diagnostics
    try:
        cache_diag = make_turboquant_cache(bit_width=8, mode="mse", per_channel_norm=False)
        with torch.no_grad():
            model(input_ids, past_key_values=cache_diag, use_cache=True)
        l0 = cache_diag.layers[0]
        k_norm = l0.keys.norm(dim=-1).mean().item()
        v_norm = l0.values.norm(dim=-1).mean().item()
        log(f"  K norm: {k_norm:.1f}, V norm: {v_norm:.1f}, ratio: {k_norm/max(v_norm,0.01):.1f}x")
        del cache_diag
    except Exception as e:
        k_norm, v_norm = -1, -1
        log(f"  Could not get KV norms: {e}")

    # Test configurations
    log(f"\n  {'K':>3s} {'V':>3s} {'avg':>5s} {'PPL':>8s} {'dPPL%':>8s} {'time':>6s}")
    log(f"  {'-'*40}")

    model_results = {
        "name": model_name,
        "params_m": params_m,
        "info": info,
        "baseline_ppl": ppl_base,
        "k_norm": k_norm,
        "v_norm": v_norm,
        "configs": [],
    }

    for kb, vb in BIT_CONFIGS:
        gc.collect()
        try:
            t0 = time.time()
            cache = make_turboquant_cache(
                key_bits=kb, value_bits=vb, seed=42,
                mode="mse", per_channel_norm=False
            )
            ppl = eval_ppl(model, input_ids, cache)
            elapsed = time.time() - t0
            diff_pct = (ppl - ppl_base) / ppl_base * 100
            avg_bits = (kb + vb) / 2
            verdict = "LOSSLESS" if abs(diff_pct) < 1 else \
                      "GOOD" if abs(diff_pct) < 5 else \
                      "OK" if abs(diff_pct) < 15 else "BAD"
            log(f"  {kb:>3d} {vb:>3d} {avg_bits:>5.1f} {ppl:>8.2f} {diff_pct:>+8.1f}% {elapsed:>5.1f}s {verdict}")
            model_results["configs"].append({
                "k_bits": kb, "v_bits": vb, "avg_bits": avg_bits,
                "ppl": ppl, "diff_pct": diff_pct, "verdict": verdict,
            })
        except Exception as e:
            log(f"  {kb:>3d} {vb:>3d}  ERROR: {e}")

    results.append(model_results)

    # Cleanup
    del model
    gc.collect()
    log(f"  Model unloaded.")


def write_summary(results: list):
    """Write summary markdown file."""
    lines = [
        "# TurboQuant Auto Benchmark — Progressive Model Testing",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Models tested: {len(results)}",
        "",
        "---",
        "",
    ]

    for r in results:
        info = r["info"]
        lines.append(f"## {r['name']} ({r['params_m']:.0f}M)")
        lines.append(f"")
        lines.append(f"- Architecture: {info['layers']}L, {info['heads']}H ({info['kv_heads']}KV), head_dim={info['head_dim']}")
        lines.append(f"- Baseline PPL: {r['baseline_ppl']:.2f}")
        lines.append(f"- K norm: {r['k_norm']:.1f}, V norm: {r['v_norm']:.1f}")
        lines.append(f"")
        lines.append(f"| K | V | Avg bits | PPL | dPPL% | Verdict |")
        lines.append(f"|---|---|----------|-----|-------|---------|")
        for c in r["configs"]:
            lines.append(
                f"| {c['k_bits']} | {c['v_bits']} | {c['avg_bits']:.1f} | "
                f"{c['ppl']:.2f} | {c['diff_pct']:+.1f}% | {c['verdict']} |"
            )
        lines.append(f"")

    # Summary table
    lines.extend(["---", "", "## Best Configuration Per Model", "",
                   "| Model | Params | Best Config | Avg bits | dPPL% |",
                   "|-------|--------|-------------|----------|-------|"])
    for r in results:
        # Find best config with <5% degradation
        good = [c for c in r["configs"] if abs(c["diff_pct"]) < 5]
        if good:
            best = min(good, key=lambda c: c["avg_bits"])
            lines.append(
                f"| {r['name']} | {r['params_m']:.0f}M | "
                f"K={best['k_bits']}/V={best['v_bits']} | {best['avg_bits']:.1f} | "
                f"{best['diff_pct']:+.1f}% |"
            )
        else:
            lines.append(f"| {r['name']} | {r['params_m']:.0f}M | No good config | - | - |")

    lines.extend(["", "---",
                   f"", f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Syn-claude*"])

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Summary written to {RESULTS_PATH}")


def main():
    # Clear log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    log("TurboQuant Auto Benchmark — Starting progressive model testing")
    log(f"Models to test: {len(MODELS)}")
    log(f"Bit configs: {len(BIT_CONFIGS)}")
    log(f"RAM: ~64GB, GPU: AMD RX 6750 XT (not used, CPU-only torch)")
    log("")

    results = []

    for i, model_name in enumerate(MODELS):
        log(f"\n[{i+1}/{len(MODELS)}] Next model: {model_name}")

        # Check model accessibility first
        info = get_model_info(model_name)
        if info is None:
            log(f"  SKIP: Cannot access model config")
            continue

        # Estimate memory requirement
        # Rough: hidden_size^2 * num_layers * 4 bytes * 4 (weights) / 1e9
        est_gb = info["hidden"] ** 2 * info["layers"] * 16 / 1e9
        log(f"  Estimated memory: ~{est_gb:.1f} GB")

        if est_gb > 50:  # Leave headroom for KV cache and OS
            log(f"  SKIP: Too large for 64GB RAM (estimated {est_gb:.1f} GB)")
            continue

        try:
            test_model(model_name, results)
        except Exception as e:
            log(f"  FATAL: {e}")
            import traceback
            log(traceback.format_exc())

        # Write intermediate summary after each model
        write_summary(results)
        gc.collect()

    log("\n" + "=" * 60)
    log("AUTO BENCHMARK COMPLETE")
    log(f"Models tested: {len(results)}")
    log(f"Results: {RESULTS_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    main()
