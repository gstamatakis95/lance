"""Quick mixed precision benchmark — single script, clean process."""
import gc
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.kv_cache import make_turboquant_cache

TEXT = (
    "In 1905, Albert Einstein published four groundbreaking papers that "
    "fundamentally changed physics. The photoelectric effect paper proposed "
    "that light consists of discrete quanta, which earned him the Nobel Prize "
    "in 1921. His paper on Brownian motion provided empirical evidence for the "
    "existence of atoms. The special relativity paper introduced the concept "
    "that the laws of physics are the same for all non-accelerating observers. "
    "The development of quantum mechanics in the 1920s was equally revolutionary. "
    "Werner Heisenberg formulated matrix mechanics in 1925."
)


def ppl(model, input_ids, cache=None):
    with torch.no_grad():
        if cache is not None:
            o = model(input_ids, past_key_values=cache, use_cache=True)
        else:
            o = model(input_ids, use_cache=False)
    l = o.logits[:, :-1, :]
    t = input_ids[:, 1:]
    return math.exp(torch.nn.CrossEntropyLoss()(
        l.reshape(-1, l.size(-1)), t.reshape(-1)
    ).item())


def main():
    print("Loading Qwen2.5-1.5B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float32)
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(TEXT)])
    print(f"Tokens: {input_ids.shape[1]}", flush=True)

    base = ppl(model, input_ids)
    print(f"Baseline PPL: {base:.2f}", flush=True)
    print(flush=True)

    configs = [
        ("uniform K=8 V=4", dict(key_bits=8, value_bits=4, mode="mse")),
        ("uniform K=6 V=3", dict(key_bits=6, value_bits=3, mode="mse")),
        ("mixed  K=3 V=4", dict(key_bits=3, value_bits=4, mode="mixed", outlier_threshold=3.0, max_outlier_ratio=0.20)),
        ("mixed  K=3 V=3", dict(key_bits=3, value_bits=3, mode="mixed", outlier_threshold=3.0, max_outlier_ratio=0.20)),
        ("mixed  K=2 V=4", dict(key_bits=2, value_bits=4, mode="mixed", outlier_threshold=3.0, max_outlier_ratio=0.20)),
        ("mixed  K=2 V=3", dict(key_bits=2, value_bits=3, mode="mixed", outlier_threshold=3.0, max_outlier_ratio=0.20)),
    ]

    for label, kwargs in configs:
        gc.collect()
        c = make_turboquant_cache(seed=42, **kwargs)
        p = ppl(model, input_ids, c)
        diff = (p - base) / base * 100
        avg_k = c.layers[0]._avg_key_bits if c.layers else kwargs.get("key_bits", 4)
        avg_v = kwargs.get("value_bits", 4)
        avg = (avg_k + avg_v) / 2
        print(f"  {label}: avg={avg:.1f}b PPL={p:.2f} ({diff:+.1f}%)", flush=True)

    del model
    gc.collect()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
