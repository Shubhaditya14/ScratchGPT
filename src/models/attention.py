import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, seq_len: int):
        super().__init__()

        assert n_embd % n_head == 0, "Embedding dim must divide evenly into heads."

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.seq_len = seq_len

        # One fused projection for Q, K, V (GPT-2 style)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # Output projection (mix heads)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Precompute the causal mask once
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("bias", mask.view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.size()   # batch, seq_len, embedding_dim

        # One linear for Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape into heads
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Compute attention scores (QKᵀ)
        att = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)

        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax -> attention probabilities
        att = F.softmax(att, dim=-1)

        # Weighted sum of values
        out = att @ v  # (B, n_head, T, head_size)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.c_proj(out)

        return out
