A minimal GPT-2 (124M) implementation built from scratch in PyTorch.

ScratchGPT is an educational implementation of GPT-2, focused on understanding transformers end-to-end: tokenization, attention, transformer blocks, training loops, and autoregressive generation.
The codebase is lightweight, readable, and easy to extend.

📦 Features

GPT-2 Small architecture (124M params)

GPT-2 BPE tokenization (tiktoken)

Multi-head self-attention

Causal masking

Transformer blocks (MHSA + MLP + residuals + LayerNorm)

AdamW optimizer with warmup + cosine decay

Dataloaders for packed sequences

Text generation (temperature, top-k, top-p)

TPU v5e support (XLA multi-core training)

📁 Project Structure
ScratchGPT/
  model/
    gpt.py
    block.py
    attention.py
  data/
    loader.py
    shakespeare.txt
  tokenizer.py
  config.py
  scripts/
    train_gpu.py
    train_tpu_v5e.py
    generate.py

🔧 Training
GPU
pip install -r requirements.txt
python -m scripts.train_gpu

TPU (v5e)
python -m scripts.train_tpu_v5e

✨ Generate Text
python -m scripts.generate --checkpoint checkpoints/gpt2_124m.pt

📘 Concepts Learned

GPT-2 BPE tokenization

Shifted (x, y) next-token prediction

Embeddings + positional encodings

Q/K/V projections and attention scores

Multi-head attention + concatenation

Feed-forward MLP layers

Residual connections + LayerNorm

Causal masking

Training loop construction

Sampling for text generation
