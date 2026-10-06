import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from model.gpt import GPT, GPTConfig
from data.loader import DataLoader
from tokenizer import encode

import os


def train_loop(index):

    device = xm.xla_device()
    xm.master_print(f"Core {index} using device: {device}")

    # Hyperparameters
    batch_size = 32    # effective batch = 32 * 8 cores = 256
    seq_len = 128
    base_lr = 3e-4
    max_steps = 20000

    # Load dataset (only once on CPU)
    if index == 0:
        xm.master_print("Loading dataset...")
    with open("data/shakespeare.txt", "r") as f:
        text = f.read()
    tokens = encode(text)

    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_loader = DataLoader(train_tokens, batch_size, seq_len)
    val_loader = DataLoader(val_tokens, batch_size, seq_len)

    # Model
    config = GPTConfig(
        vocab_size=50257,
        n_embd=256,
        n_head=8,
        n_layer=8,
        seq_len=seq_len,
        dropout=0.0
    )

    model = GPT(config).to(device)
    model.lm_head.weight = model.wte.weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # Make dataloader parallel across TPU cores
    train_pl = pl.MpDeviceLoader(train_loader, device)
    val_pl = pl.MpDeviceLoader(val_loader, device)

    steps = 0

    for batch in train_pl:
        xb, yb = batch
        xb = xb.to(device)
        yb = yb.to(device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)

        if index == 0 and steps % 50 == 0:
            xm.master_print(f"Step {steps} | Loss {loss.item():.4f}")

        if steps >= max_steps:
            break

        steps += 1

    if index == 0:
        xm.master_print("Saving checkpoint...")
        xm.save(model.state_dict(), "mini_gpt_parallel.pt")


def main():
    # Launch on 8 TPU cores
    xmp.spawn(train_loop, nprocs=8)


if __name__ == "__main__":
    main()
