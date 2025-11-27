import torch.nn as nn
from .attention import MultiHeadSelfAttention


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc_in = nn.Linear(n_embd, 4 * n_embd)   # expansion
        self.act = nn.GELU()                         # GPT-2 uses GELU (approx)
        self.fc_out = nn.Linear(4 * n_embd, n_embd)  # projection back down

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act(x)
        x = self.fc_out(x)
        return x


class Block(nn.Module):
    """
    One GPT-style Transformer block:
        x = x + Attn(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self, n_embd: int, n_head: int, seq_len: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd=n_embd, n_head=n_head, seq_len=seq_len)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # Pre-LN + self-attention + residual
        x = x + self.attn(self.ln_1(x))
        # Pre-LN + MLP + residual
        x = x + self.mlp(self.ln_2(x))
        return x
