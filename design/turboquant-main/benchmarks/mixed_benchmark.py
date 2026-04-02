"""Benchmark mixed precision vs uniform quantization on Qwen2.5-1.5B."""

import gc
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from turboquant.kv_cache import make_turboquant_cache

TEXT = (
    "In 1905, Albert Einstein published four groundbreaking papers that "
    "fundamentally changed physics. The photoelectric effect paper proposed "
    "that light consists of discrete quanta, which earned him the Nobel Prize "
    "in 1921. His paper on Brownian motion provided empirical evidence for the "
    "existence of atoms. The special relativity paper introduced the concept "
    "that the laws of physics are the same for all non-accelerating observers. "
    "The development of quantum mechanics in the 1920s was equally revolutionary. "
    "Werner Heisenberg formulated matrix mechanics in 1925, while Erwin "
    "Schrodinger developed wave mechanics in 1926."
)


def eval_ppl(model, input_ids, cache=None):
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


def run(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    tokens = tokenizer.encode(TEXT)
    input_ids = torch.tensor([tokens])
    print(f"Tokens: {len(tokens)}")

    ppl_base = eval_ppl(model, input_ids)
    print(f"Baseline PPL: {ppl_base:.2f}\n")

    header = f"{'Mode':>14s} {'K':>4s} {'V':>4s} {'avg':>6s} {'PPL':>8s} {'dPPL%':>8s}"
    print(header)
    print("-" * len(header))

    configs = [
        # (label, mode, key_bits, value_bits, threshold)
        ("uniform", "mse", 8, 4, None),
        ("uniform", "mse", 8, 3, None),
        ("uniform", "mse", 6, 4, None),
        ("uniform", "mse", 6, 3, None),
        ("mixed t=3", "mixed", 4, 4, 3.0),
        ("mixed t=3", "mixed", 3, 4, 3.0),
        ("mixed t=3", "mixed", 3, 3, 3.0),
        ("mixed t=3", "mixed", 2, 4, 3.0),
        ("mixed t=3", "mixed", 2, 3, 3.0),
        ("mixed t=2", "mixed", 4, 4, 2.0),
        ("mixed t=2", "mixed", 3, 4, 2.0),
        ("mixed t=2", "mixed", 3, 3, 2.0),
        ("mixed t=2", "mixed", 2, 4, 2.0),
        ("mixed t=2", "mixed", 2, 3, 2.0),
    ]

    for label, mode, kb, vb, thresh in configs:
        gc.collect()
        kwargs = {"key_bits": kb, "value_bits": vb, "mode": mode, "seed": 42}
        if thresh is not None:
            kwargs["outlier_threshold"] = thresh
        try:
            cache = make_turboquant_cache(**kwargs)
            ppl = eval_ppl(model, input_ids, cache)
            diff = (ppl - ppl_base) / ppl_base * 100
            # Get actual avg bits
            if cache.layers and hasattr(cache.layers[0], "_avg_key_bits"):
                avg_k = cache.layers[0]._avg_key_bits
            else:
                avg_k = kb
            avg = (avg_k + vb) / 2
            print(f"{label:>14s} {kb:>4d} {vb:>4d} {avg:>6.1f} {ppl:>8.2f} {diff:>+8.1f}%")
        except Exception as e:
            print(f"{label:>14s} {kb:>4d} {vb:>4d}  ERROR: {e}")

    del model
    gc.collect()


if __name__ == "__main__":
    run("Qwen/Qwen2.5-1.5B")
