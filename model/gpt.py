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
