---
title: GPT-2 Like Model
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
---

# Small-Scale GPT-2-Style Language Model

A decoder-only Transformer language model implemented from scratch in PyTorch and trained on WikiText-103 with a custom 32K BPE tokenizer. The project covers the full pipeline: tokenizer training, model architecture, training with resumable checkpoints, KV-cached inference with three decoding strategies, an interactive Gradio demo, Docker packaging, and CI on GitHub Actions.

## Highlights

- **From-scratch architecture.** Multi-head causal self-attention, pre-LayerNorm blocks, GELU feed-forward layers, learned positional embeddings, and tied input/output embeddings. No `nn.Transformer` or pretrained weights.
- **Custom tokenizer.** Byte-pair encoding with a 32,000-token vocabulary trained on WikiText-103 using Hugging Face `tokenizers`.
- **Training.** AdamW with decoupled weight decay, linear warmup followed by cosine decay, gradient clipping, best-on-validation checkpoint selection, and resumable training.
- **Efficient inference.** A key/value cache cuts per-token decoding latency by up to **5.4×** on CPU and produces exactly the same output as full recomputation.
- **Decoding.** Greedy, top-k, and nucleus (top-p) sampling with temperature, a repetition penalty, and special-token suppression.
- **Engineering.** Gradio demo, Docker image, and a unit and integration test suite that runs on every push and pull request.

## Results

| Metric | Value |
|---|---|
| Test perplexity (WikiText-103) | **33.07** |
| Parameters | 33.7M |
| Training steps | 200K (~0.8B tokens) |
| KV-cache decoding speedup (CPU, 248 new tokens) | **5.4×** |

## Model

| Hyperparameter | Value |
|---|---|
| Layers | 12 |
| Hidden size (`d_model`) | 384 |
| Attention heads | 6 (head dim 64) |
| Feed-forward size (`d_ff`) | 1536 |
| Context length | 256 tokens |
| Vocabulary | 32,000 (BPE) |
| Dropout | 0.1 |

Hyperparameters live in [`config.py`](config.py).

## Training

| Setting | Value |
|---|---|
| Dataset | WikiText-103 (raw), loaded with Hugging Face `datasets` |
| Optimizer | AdamW, lr 2e-4, weight decay 0.01 (none on biases or LayerNorm) |
| Schedule | 3,000 warmup steps, then cosine decay to 1% of peak |
| Batch size | 16 sequences × 256 tokens |
| Steps | 20 epochs × 10,000 steps |
| Gradient clipping | Global norm 1.0 |
| Checkpointing | Latest state in `checkpoints/latest.pt` (resumes automatically); best validation model in `best_gpt_model.pt` |

After training, the best validation checkpoint is evaluated on the test split. Test perplexity is `exp(mean cross-entropy loss)`.

## Inference with a KV Cache

Without a cache, every new token requires a forward pass over the entire sequence, so the keys and values of earlier tokens are recomputed at every step. With a cache, each attention layer stores its keys and values, and each step processes only the newest token.

Implementation notes:

- `TransformerModel.forward(input_ids, past_kvs=None, use_cache=False)` returns `(logits, present_kvs)` when `use_cache=True`. The default call path, which training uses, is unchanged.
- Positional embeddings start at the cached length (`start_pos`), and the causal mask is offset so that new tokens can attend to every cached position.
- Positions are absolute. When a generated sequence grows past the 256-token context window, the window slides and the cache is rebuilt from the cropped context.
- `TextGenerator(..., use_cache=True)` is the default. Pass `use_cache=False` to compare against full recomputation.

### Benchmark

These numbers come from greedy decoding in float32 with batch size 1 and an 8-token prompt. Each result is the fastest of 3 runs. The model uses the configuration above with random weights, because weights do not affect speed. Hardware: Apple Silicon MacBook Air.

**CPU**

| New tokens | No cache (ms/token) | KV cache (ms/token) | Speedup | Cache size |
|---:|---:|---:|---:|---:|
| 32 | 18.4 | 6.3 | 2.9× | 1.4 MB |
| 64 | 20.5 | 5.5 | 3.7× | 2.5 MB |
| 128 | 26.7 | 5.6 | 4.8× | 4.8 MB |
| 248 | 42.9 | 8.0 | **5.4×** | 9.0 MB |

**Apple GPU (MPS)**

| New tokens | No cache (ms/token) | KV cache (ms/token) | Speedup |
|---:|---:|---:|---:|
| 32 | 33.3 | 39.7 | 0.84× |
| 64 | 38.0 | 47.0 | 0.81× |
| 128 | 63.5 | 64.5 | 0.98× |
| 248 | 164.9 | 90.3 | **1.8×** |

Observations:

- On CPU, the uncached cost per token grows with sequence length, while the cached cost stays roughly flat. The speedup therefore increases with output length.
- On MPS, single-token steps are too small to use the GPU efficiently, so kernel-launch overhead dominates. The cache helps only for longer outputs. The Hugging Face Space runs on CPU, where the cache pays off.
- Cache memory is `2 × layers × seq_len × d_model × 4 bytes`, which is 9 MB at full context for this model.
- The cached and uncached outputs are identical. The test suite verifies this (`KVCacheTests`).

To reproduce:

```bash
python benchmark_kv_cache.py --device cpu
python benchmark_kv_cache.py --device mps            # or cuda
python benchmark_kv_cache.py --checkpoint gpt_model.pt
```

## Getting Started

### Install

```bash
git clone https://github.com/franciscochen435/based-on-GPT-2-small-scale-Language-model.git
cd based-on-GPT-2-small-scale-Language-model
pip install torch tokenizers datasets matplotlib gradio pytest
```

### Train

```bash
python TransformerTrainer.py
```

WikiText-103 downloads automatically. Training resumes from `checkpoints/latest.pt` if that file exists. When training finishes, the script writes `best_gpt_model.pt`, `gpt_model.pt`, and `learning_curve.png`. A GPU is strongly recommended, and Google Colab works.

### Generate Text

```bash
python generate_samples.py
```

This script loads `gpt_model.pt`, runs greedy, top-k, and nucleus decoding on a set of prompts, and writes `generated_samples.json` and `generated_samples.txt`.

### Run the Demo

```bash
python app.py                      # http://localhost:7860
```

Or run it with Docker:

```bash
docker build -t gpt2-small .
docker run -p 7860:7860 gpt2-small
```

The demo lists every `*.pt` checkpoint in the project root and in `models/`.

### Test

```bash
pytest -q test_project.py
```

## Project Structure

```
├── transformer/
│   ├── Embedding.py            # token + learned positional embeddings
│   ├── SelfAttention.py        # multi-head causal self-attention (with KV cache)
│   ├── FeedForward.py          # position-wise MLP (GELU)
│   ├── TransformerBlock.py     # pre-LN residual block
│   ├── TransformerModel.py     # full decoder-only model
│   └── TransformerBuilder.py   # Builder for model construction
├── tokenizer/                  # BPE tokenizer training scripts and trained tokenizer
├── BaseTokenizer.py            # tokenizer interface
├── HFTokenizerAdapter.py       # Adapter over Hugging Face tokenizers
├── TokenizerFactory.py         # Factory for tokenizer backends
├── BaseTrainer.py              # trainer interface (template method: run())
├── TransformerTrainer.py       # data loading, optimization, evaluation
├── TrainingObserver.py         # Observer-based training progress reporting
├── checkpoint.py               # save/load resumable checkpoints
├── dataset.py                  # sliding-window language-modeling dataset
├── generate_samples.py         # TextGenerator: greedy / top-k / nucleus decoding
├── benchmark_kv_cache.py       # KV cache latency benchmark
├── app.py                      # Gradio demo
├── test_project.py             # unit and integration tests
├── config.py                   # hyperparameters
├── Dockerfile
└── .github/workflows/ci-cd.yml
```

The codebase uses standard design patterns to keep components decoupled. A **Builder** constructs the model. A **Factory** and an **Adapter** let tokenizer backends be swapped. **Observers** report training progress, so logging is separate from the training loop. `BaseTrainer` defines the training workflow, and concrete trainers fill in each step.

## Continuous Integration

GitHub Actions runs on every push and pull request to `main`. It installs dependencies, byte-compiles all sources, and runs the full test suite. The suite covers each module's shape contracts and input validation, weight tying, the model builder, the learning-rate schedule, checkpoint round-trips, end-to-end training steps, and KV-cache equivalence.
