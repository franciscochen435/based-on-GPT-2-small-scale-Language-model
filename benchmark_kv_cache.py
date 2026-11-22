"""Benchmark greedy generation with and without the KV cache.

Measures the model alone (no tokenizer), using the architecture in config.py.
Weights do not affect speed, so a checkpoint is optional.

Examples:
    python benchmark_kv_cache.py                      # auto device, random weights
    python benchmark_kv_cache.py --device cpu
    python benchmark_kv_cache.py --device cuda --checkpoint gpt_model.pt
    python benchmark_kv_cache.py --device mps --dtype float16
"""
import argparse
import csv
import platform
import time

import torch

from config import vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff
from transformer.TransformerBuilder import TransformerModelBuilder


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_model(device, dtype, checkpoint):
    model = (
        TransformerModelBuilder()
        .with_vocab_size(vocab_size).with_max_seq_len(max_seq_len)
        .with_d_model(d_model).with_n_heads(n_heads).with_n_layers(n_layers)
        .with_d_ff(d_ff).with_dropout(0.0)
        .build()
    )
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def generate(model, prompt, new_tokens, use_cache):
    ids = prompt
    cache = None
    for _ in range(new_tokens):
        if use_cache:
            step_in = ids if cache is None else ids[:, -1:]
            logits, cache = model(step_in, past_kvs=cache, use_cache=True)
        else:
            logits = model(ids)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids


def time_run(model, prompt, new_tokens, use_cache, device, repeats):
    generate(model, prompt, min(new_tokens, 8), use_cache)  # warm-up
    best = float("inf")
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        generate(model, prompt, new_tokens, use_cache)
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--new-tokens", default="32,64,128,248",
                    help="comma-separated; prompt-len + new tokens must be <= max_seq_len")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None, help="CSV path (default: kv_bench_<device>.csv)")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    model = build_model(device, dtype, args.checkpoint)
    prompt = torch.randint(0, vocab_size, (1, args.prompt_len), device=device)
    lengths = [int(x) for x in args.new_tokens.split(",")]
    assert args.prompt_len + max(lengths) <= max_seq_len, "sequence would exceed max_seq_len"

    # Sanity check: both paths must produce the same tokens.
    a = generate(model, prompt, 16, use_cache=False)
    b = generate(model, prompt, 16, use_cache=True)
    print("outputs identical with/without cache:", torch.equal(a, b))

    hw = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine()
    print(f"device={device} ({hw})  dtype={args.dtype}  layers={n_layers} d_model={d_model}")
    bytes_per = torch.finfo(dtype).bits // 8
    rows = []
    print(f"{'new_tokens':>10} {'no_cache_s':>11} {'cache_s':>9} {'speedup':>8} "
          f"{'ms/tok_no':>10} {'ms/tok_yes':>11} {'cache_MB':>9}")
    for n in lengths:
        t_no = time_run(model, prompt, n, False, device, args.repeats)
        t_yes = time_run(model, prompt, n, True, device, args.repeats)
        total = args.prompt_len + n
        cache_mb = 2 * n_layers * total * d_model * bytes_per / 2**20  # K and V, batch 1
        row = dict(device=device, hardware=hw, dtype=args.dtype, new_tokens=n,
                   no_cache_s=round(t_no, 4), cache_s=round(t_yes, 4),
                   speedup=round(t_no / t_yes, 2),
                   ms_per_token_no_cache=round(1000 * t_no / n, 3),
                   ms_per_token_cache=round(1000 * t_yes / n, 3),
                   kv_cache_mb=round(cache_mb, 2))
        rows.append(row)
        print(f"{n:>10} {t_no:>11.3f} {t_yes:>9.3f} {t_no / t_yes:>7.2f}x "
              f"{row['ms_per_token_no_cache']:>10.2f} {row['ms_per_token_cache']:>11.2f} {cache_mb:>9.2f}")

    out = args.out or f"kv_bench_{device}_{args.dtype}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
