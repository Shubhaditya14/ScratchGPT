# ScratchGPT

A minimal GPT-2 implementation built from scratch in PyTorch.

ScratchGPT is an educational implementation of GPT-2 focused on understanding transformers end to end, from tokenization and attention to training and autoregressive generation.

The codebase is lightweight, readable, and designed to be easy to extend.

## Features

* GPT-2 Small architecture
* GPT-2 BPE tokenization with `tiktoken`
* Multi-head self-attention
* Causal masking
* Transformer blocks with MHSA, MLPs, residual connections, and LayerNorm
* AdamW optimizer
* Warmup + cosine learning rate decay
* Packed sequence dataloaders
* Autoregressive text generation
* Temperature, top-k, and top-p sampling

## Project Structure

```text
ScratchGPT/
├── model/
│   ├── gpt.py
│   ├── block.py
│   └── attention.py
├── data/
│   ├── loader.py
│   └── shakespeare.txt
├── tokenizer.py
├── config.py
├── scripts/
│   ├── train_gpu.py
│   └── generate.py
└── requirements.txt
```

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m scripts.train_gpu
```

## Generate Text

Generate text from a trained checkpoint:

```bash
python -m scripts.generate --checkpoint checkpoints/gpt2.pt
```

## Concepts Learned

* GPT-2 BPE tokenization
* Shifted next-token prediction
* Token and positional embeddings
* Query, Key, and Value projections
* Attention scores and scaling
* Multi-head self-attention
* Causal masking
* Feed-forward MLP layers
* Residual connections
* Layer normalization
* Transformer block construction
* Training loop design
* AdamW optimization
* Learning rate scheduling
* Autoregressive text generation
* Temperature, top-k, and top-p sampling

## Goal

ScratchGPT is primarily a learning project: a compact implementation for understanding how GPT style language models work internally rather than relying entirely on high level abstractions.
