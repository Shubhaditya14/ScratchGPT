# ScratchGPT

A minimal GPT-2 implementation from scratch with Shakespeare data.

## Quick Start

```bash
# Run the interactive TUI
./run.zsh

# Or just generate text (run from the project root)
python -m src.scripts.generate

# Check if model is trained (recommended first step!)
python -m src.scripts.diagnose
```

## Project Structure

```
src/
  models/       # GPT model (gpt.py, attention.py, block.py)
  scripts/      # train.py, generate.py, evaluate.py, diagnose.py, tui.py, train_tpu.py
  data/         # shakespeare.txt dataset, loader.py (batching)
  tokenizer.py  # GPT-2 tokenizer wrapper
checkpoints/    # Saved model weights
run.zsh         # Entry point script
```

## Training

```bash
python -m src.scripts.train
```

Hyperparameters:
- batch_size: 8
- seq_len: 128
- learning_rate: 3e-4
- max_iters: 2000
- warmup_steps: 400 (linear warmup, then cosine decay to 1e-5)
- eval_interval: 200

The data is split 90/10 into train/val. TensorBoard logs go to `runs/mini_gpt`, and the
checkpoint is saved to `checkpoints/mini_gpt.pt` when training finishes.

To measure validation loss on a saved checkpoint:
```bash
python -m src.scripts.evaluate
```

## Generation

**Generate text** (non-interactive):
```bash
python -m src.scripts.generate
```

**Interactive TUI** (chat with model):
```bash
./run.zsh
```

Commands in TUI:
- `/help` - Show all commands
- `/temp <float>` - Set temperature (0.8 is default, sweet spot)
- `/topk <int|none>` - Set top-k filtering (40 is default, `none` or `0` disables it)
- `/max <int>` - Max tokens to generate (200 is default)
- `/ckpt <path>` - Load different checkpoint
- `/clear` - Clear output
- `/quit` or `/exit` - Exit

## Sampling Parameters Explained

### Temperature
Controls randomness of token selection:
- `temp = 0.0` → Always pick most likely (greedy) → Repetition collapse ❌
- `temp = 0.8` → Sweet spot ✓ (default)
- `temp = 1.0` → Sample from the model's unscaled distribution (within top-k)
- `temp > 1.5` → Very random, low coherence ❌

### Top-K Filtering
Only sample from the K most likely tokens:
- `top_k = None` → No filtering (can lead to nonsense)
- `top_k = 40` → Sweet spot ✓ (default)
- `top_k = 50` → Slightly more diversity
- `top_k = 20` → More deterministic

**Default combo (0.8 temp + 40 top_k)** is best for coherent output.

## Troubleshooting

### "Repetition Collapse" (same words repeating)
This means the model is picking the same high-probability token over and over.

**Fixes:**
1. Lower temperature: `/temp 0.6`
2. Increase top-k: `/topk 50`
3. Check if model is trained: `python -m src.scripts.diagnose`

### Model not trained?
```bash
# Check validation loss
python -m src.scripts.diagnose

# If val_loss > 3.5, model needs more training
python -m src.scripts.train
```

### Import errors?
The scripts import from the `src` package, so run them as modules from the project root
(`python src/scripts/generate.py` fails with `No module named 'src'`):
```bash
cd /path/to/gpt2
python -m src.scripts.generate
```

## Model Config

```python
vocab_size: 50257      # GPT-2 vocab
n_embd: 128            # Embedding dim
n_head: 4              # Number of attention heads
n_layer: 4             # Number of transformer layers
seq_len: 128           # Max sequence length
dropout: 0.0           # Accepted by GPTConfig but not applied (no dropout layers)
```

Very small model (~7.2M params with tied input/output embeddings) for fast training on CPU/MPS. For better quality:
- Increase `n_embd` to 256 or 512
- Increase `n_layer` to 6 or 8
- Use more training data

## References

- GPT-2 paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
- nanoGPT: https://github.com/karpathy/nanoGPT
- Top-K sampling: https://arxiv.org/abs/1805.04833
