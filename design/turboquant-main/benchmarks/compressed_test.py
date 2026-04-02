"""Test compressed KV cache with actual memory savings."""
import gc
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.compressed_cache import make_compressed_cache


def main():
    text = (
        "In 1905, Albert Einstein published four groundbreaking papers. "
        "The photoelectric effect paper proposed that light consists of "
        "discrete quanta. His paper on Brownian motion provided evidence. "
        "The special relativity paper was also revolutionary."
    )

    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(text)])
    n = input_ids.shape[1]
    print(f"Tokens: {n}", flush=True)

    # Baseline
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    base_ppl = math.exp(
        torch.nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        ).item()
    )
    print(f"Baseline PPL: {base_ppl:.2f}", flush=True)

    # Compressed cache
    gc.collect()
    print("Running compressed cache...", flush=True)
    cache = make_compressed_cache(key_bits=3, value_bits=3, outlier_threshold=3.0)
    with torch.no_grad():
        out2 = model(input_ids, past_key_values=cache, use_cache=True)
    logits2 = out2.logits[:, :-1, :]
    comp_ppl = math.exp(
        torch.nn.CrossEntropyLoss()(
            logits2.reshape(-1, logits2.size(-1)), labels.reshape(-1)
        ).item()
    )
    diff = (comp_ppl - base_ppl) / base_ppl * 100
    print(f"Compressed PPL: {comp_ppl:.2f} ({diff:+.1f}%)", flush=True)

    # Memory stats
    total_orig = 0
    total_comp = 0
    for layer in cache.layers:
        s = layer.memory_stats()
        total_orig += s["original_bytes"]
        total_comp += s["compressed_bytes"]

    print(flush=True)
    print(f"=== Memory Savings ({len(cache.layers)} layers, {n} tokens) ===", flush=True)
    print(f"  FP32 (baseline): {total_orig:>10,} bytes ({total_orig/1024:.1f} KB)", flush=True)
    print(f"  Compressed:      {total_comp:>10,} bytes ({total_comp/1024:.1f} KB)", flush=True)
    print(f"  Saved:           {total_orig-total_comp:>10,} bytes ({(total_orig-total_comp)/1024:.1f} KB)", flush=True)
    print(f"  Ratio:           {total_orig/max(total_comp,1):.1f}x", flush=True)
    print(f"  Reduction:       {(1-total_comp/total_orig)*100:.0f}%", flush=True)

    del model
    gc.collect()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
