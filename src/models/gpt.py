import torch
import torch.nn as nn
from .block import Block


class GPTConfig:
    def __init__(self,
                 vocab_size: int = 50257,
                 n_embd: int = 128,
                 n_head: int = 4,
                 n_layer: int = 4,
                 seq_len: int = 128,
                 dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.dropout = dropout

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # positional embeddings
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config.n_embd, config.n_head, config.seq_len)
            for _ in range(config.n_layer)
        ])

        # final layernorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM head (tied to token embeddings later)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
            B, T = idx.shape  # batch size, sequence length

            # Make sure input isn't longer than model's max sequence length
            assert T <= self.config.seq_len, "Sequence length exceeds model capacity."

            # Position indices [0, 1, 2, ..., T-1]
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

            # Token and position embeddings
            tok_emb = self.wte(idx)        # (B, T, n_embd)
            pos_emb = self.wpe(pos)        # (T, n_embd)

            # Add embeddings
            x = tok_emb + pos_emb

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            # Final LayerNorm
            x = self.ln_f(x)

            # Logits over vocabulary
            logits = self.lm_head(x)       # (B, T, vocab_size)

            # If no labels given → just return logits (useful for inference)
            if targets is None:
                return logits

            # Flatten logits and targets for cross-entropy
            logits_flat = logits.view(-1, self.config.vocab_size)
            targets_flat = targets.view(-1)

            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

            return logits, loss


