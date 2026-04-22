---
title: GPT-2 Small-Scale Language Model
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
---

# Small-Scale GPT-2 Language Model

This project is to build a small-scale language model inspired by GPT-2 from the ground up, while following the software engineering lifecycle, including requirements analysis, system design, implementation, testing, documentation, UML diagrams, and design patterns.

The complete project includes tokenizer development, GPT-like pre-training, text generation, classification fine-tuning, instruction tuning for question answering, deployment/API development, UML documentation, and final reporting. The current codebase mainly implements the pre-training stage: a BPE tokenizer, a GPT-style decoder-only Transformer, a training pipeline for next-token prediction, checkpointing, evaluation, and text generation.

## Features

- Uses a BPE tokenizer to convert text into token IDs.
- Implements a GPT-like decoder-only Transformer from scratch.
- Supports token embeddings, positional embeddings, causal multi-head self-attention, feed-forward layers, LayerNorm, and residual connections.
- Trains on WikiText for next-token prediction.
- Uses AdamW, weight decay, gradient clipping, and a warmup + cosine decay learning-rate schedule.
- Supports checkpoint saving and training resume.
- Computes cross-entropy loss on validation/test splits and reports perplexity from test loss.
- Saves the best model, final model, and learning curve.
- Generates samples with greedy decoding, top-k sampling, and nucleus sampling.
- Uses the Builder Pattern for model construction and the Observer Pattern for training monitoring.

## Project Structure

```text
.
├── BaseTrainer.py                  # Abstract trainer base class
├── TrainingObserver.py             # Training observer interface and console observer
├── TransformerTrainer.py           # Main training pipeline
├── checkpoint.py                   # Checkpoint save/load utilities
├── config.py                       # Model and training hyperparameters
├── dataset.py                      # Dataset wrapper for next-token prediction
├── generate_samples.py             # Text generation script
├── requirements.txt                # Python dependencies
├── tokenizer/
│   ├── config.json                 # Tokenizer configuration
│   ├── fastbpe.py                  # BPE tokenizer training with Hugging Face tokenizers
│   ├── tokenizer.py                # Simplified BPE/tokenizer experiment code
│   ├── test_tokenizer.py           # Tokenizer test script
│   └── trained_tokenizer/          # Saved tokenizer files
└── transformer/
    ├── Embedding.py                # Token embedding + positional embedding
    ├── FeedForward.py              # Transformer feed-forward network
    ├── SelfAttention.py            # Causal multi-head self-attention
    ├── TransformerBlock.py         # Single decoder block
    ├── TransformerBuilder.py       # Builder Pattern for model construction
    └── TransformerModel.py         # Main GPT-like Transformer model
```

## Model Design

The model follows a GPT-2-style decoder-only Transformer architecture for autoregressive language modeling. Given a token sequence of length `T`, the model outputs vocabulary logits at each position and is trained to predict the next token.

Main components:

1. `Embedding`
   - Token embeddings map token IDs into dense vectors.
   - Positional embeddings add order information to the sequence.

2. `TransformerBlock`
   - Uses Pre-LayerNorm for more stable training.
   - Uses causal self-attention so each position can only attend to previous positions and itself.
   - Uses residual connections to improve gradient flow.
   - Uses a GELU feed-forward network to increase model capacity.

3. `TransformerModel`
   - Stacks multiple Transformer decoder blocks.
   - Applies a final LayerNorm and language-model head to produce vocabulary logits.
   - Shares weights between the token embedding matrix and the language-model head.

Default hyperparameters are defined in `config.py`:

```python
vocab_size = 32000
max_seq_len = 256
d_model = 384
n_heads = 6
n_layers = 12
d_ff = 1536
dropout = 0.1
batch_size = 16
lr = 2e-4
weight_decay = 0.01
epochs = 20
warmup_steps = 3000
```

## Design Patterns

The current implementation includes several software design ideas required by the course project:

- Builder Pattern: `transformer/TransformerBuilder.py` constructs `TransformerModel` through chained `.with_*()` configuration methods.
- Observer Pattern: `TrainingObserver.py` defines a `TrainingMonitor` that notifies observers when training starts, an epoch ends, training finishes, and test evaluation completes.


## Installation

Python 3.10+ is recommended. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tokenizers datasets
```

Note: the current code imports `tokenizers` and `datasets`, but these packages are not listed in the existing `requirements.txt`, so they need to be installed separately.

If CUDA is available, the trainer will use GPU automatically. Otherwise, it will fall back to CPU.

If training is run on Google Colab, install the required packages in notebook cells before running the training script:

```python
!pip install torch
!pip install tokenizers
!pip install matplotlib
```

## Training

Run the main trainer:

```bash
python TransformerTrainer.py
```

The pre-training pipeline:

1. Loads the BPE tokenizer from `tokenizer/trained_tokenizer/tokenizer.json`.
2. Downloads `wikitext-103-raw-v1` from Hugging Face.
3. Encodes the train/validation/test splits into token IDs.
4. Uses `LMDataset` to create fixed-length language-modeling samples.
5. Builds the GPT-like Transformer model.
6. Trains with AdamW and a warmup + cosine decay scheduler.
7. Saves a checkpoint after each epoch to `checkpoints/latest.pt`.
8. Saves the best validation model to `best_gpt_model.pt`.
9. Evaluates the best model on the test set and computes perplexity.
10. Saves the final model to `gpt_model.pt` and the learning curve to `learning_curve.png`.

Training outputs:

```text
checkpoints/latest.pt
best_gpt_model.pt
gpt_model.pt
learning_curve.png
```

## Text Generation

After training finishes and `gpt_model.pt` exists, run:

```bash
python generate_samples.py
```

The script compares three decoding strategies:

- Greedy decoding: always selects the token with the highest probability. It is stable but less diverse.
- Top-k sampling: samples only from the top `k` most likely tokens, increasing diversity.
- Nucleus sampling: samples from the smallest token set whose cumulative probability exceeds `p`, balancing fluency and diversity.

Generated outputs are saved to:

```text
generated_samples.json
generated_samples.txt
```

## Tokenizer

The repository already includes a trained tokenizer:

```text
tokenizer/trained_tokenizer/tokenizer.json
```

To retrain the tokenizer, use `tokenizer/fastbpe.py`. The script expects a training text file inside the `tokenizer/` directory:

```bash
cd tokenizer
python fastbpe.py
```

The default output path is:

```text
tokenizer/trained_tokenizer/tokenizer.json
```

Important: after retraining the tokenizer, `vocab_size` in `config.py` must match the actual tokenizer vocabulary size. Otherwise, the trainer will raise an error.

## Dataset and Objective

The current training task is language modeling / next-token prediction. For a token sequence:

```text
x = [t0, t1, t2, ..., t126]
y = [t1, t2, t3, ..., t127]
```

The model receives `x` as input and learns to predict `y`. The loss function is cross-entropy loss.

The code uses Hugging Face `wikitext-103-raw-v1` as the source for training, validation, and test data. `dataset.py` provides `LMDataset`, which slices a continuous token stream into fixed-length training samples.

## Current Status

Implemented:

- BPE tokenizer integration.
- GPT-like Transformer architecture.
- Next-token prediction training pipeline.
- Checkpoint save/resume support.
- Validation/test loss and perplexity evaluation.
- Greedy, top-k, and nucleus text generation.
- Builder and Observer design pattern integration.

Future work:

- IMDb sentiment classification fine-tuning.
- Instruction tuning / question-answering fine-tuning.
- API or web UI deployment.
- Unit tests with pytest and end-to-end testing reports.
- More complete UML documentation, including use case, class, sequence, component, deployment, and design pattern diagrams.

## CI/CD Pipeline

This repository includes a GitHub Actions workflow at `.github/workflows/ci-cd.yml`.

The CI pipeline runs automatically on pushes and pull requests to `main` or `master`. It performs the following checks:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs the required runtime packages: `torch`, `tokenizers`, `datasets`, `matplotlib`, and `pytest`.
4. Runs Python syntax checks with `compileall`.
5. Runs a lightweight smoke test that builds a small Transformer model, creates a sample `LMDataset`, performs a forward pass, and verifies the output shape.

The CD pipeline runs after CI succeeds on pushes to `main` or `master`. Since this project does not currently deploy a web API or application, the CD step packages the project source code, tokenizer files, README, and dependency file as a downloadable GitHub Actions artifact named `small-scale-gpt2-project`.


## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'tokenizers'`

Run:

```bash
pip install tokenizers
```

### 2. `ModuleNotFoundError: No module named 'datasets'`

Run:

```bash
pip install datasets
```

### 3. `config vocab_size != tokenizer vocab size`

This means `vocab_size` in `config.py` does not match the actual vocabulary size of the tokenizer file. Make sure the correct tokenizer is being used, or update `config.py`.

### 4. `gpt_model.pt` does not exist

Train the model first:

```bash
python TransformerTrainer.py
```

Then run sample generation:

```bash
python generate_samples.py
```

## Summary

This project implements the core pre-training pipeline of a small GPT-2-style language model, including tokenization, decoder-only Transformer modeling, training, checkpointing, evaluation, and text generation. It corresponds mainly to the Week 5-6 pre-training deliverables in `project.pdf` and provides a foundation for later classification fine-tuning, instruction tuning, deployment, testing, and complete software engineering documentation.
