import torch
import random

class DataLoader:
    def __init__(self, tokens: list[int], batch_size: int, seq_len: int):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_tokens = len(tokens)

        assert self.num_tokens > seq_len, (
            f"Dataset too small: only {self.num_tokens} tokens "
            f"but seq_len={seq_len}"
        )

    def get_batch(self, split="train"):
        """
        Returns a batch of (x, y) tensors:
        - x: (batch_size, seq_len)
        - y: (batch_size, seq_len)
        """
        xs = []
        ys = []

        if split == "train":
            # random sampling
            for _ in range(self.batch_size):
                i = random.randint(0, self.num_tokens - self.seq_len - 2)
                x = self.tokens[i : i + self.seq_len]
                y = self.tokens[i + 1 : i + 1 + self.seq_len]
                xs.append(x)
                ys.append(y)

        elif split == "val":
            # deterministic sequential slicing (for stable evaluation)
            start = 0
            for _ in range(self.batch_size):
                if start + self.seq_len + 1 >= self.num_tokens:
                    start = 0  # wrap cleanly for evaluation
                x = self.tokens[start : start + self.seq_len]
                y = self.tokens[start + 1 : start + 1 + self.seq_len]
                xs.append(x)
                ys.append(y)
                start += self.seq_len  # move forward sequentially

        else:
            raise ValueError("split must be 'train' or 'val'")

        # convert lists to tensors (LongTensor is required for embeddings)
        x_tensor = torch.tensor(xs, dtype=torch.long)
        y_tensor = torch.tensor(ys, dtype=torch.long)

        return x_tensor, y_tensor
