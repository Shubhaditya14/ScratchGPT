import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from model.gpt import GPT, GPTConfig
from data.loader import DataLoader
from tokenizer import encode

import os


# ---- LR Scheduler (same as CPU/GPU version) ----
def get_lr(step, warmup_steps, max_steps, base_lr, min_lr=1e-5):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
    return min_lr + (base_lr - min_lr) * cosine


def train_tpu():

    # ---- TPU device ----
    device = xm.xla_device()
    xm.master_print(f"Using TPU device: {device}")

    # ---- Hyperparameters ----
    batch_size = 32       # TPU can handle this
    seq_len = 128
    base_lr = 3e-4
    max_iters = 20000     # TPU can easily run this
    warmup_steps = 400
    eval_interval = 500

    # ---- Load dataset ----
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = encode(text)

    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_loader = DataLoader(train_tokens, batch_size, seq_len)
    val_loader = DataLoader(val_tokens, batch_size, seq_len)

    # ---- Model config ----
    config = GPTConfig(
        vocab_size=50257,
        n_embd=256,     # larger model for TPU
        n_head=8,
        n_layer=8,
        seq_len=seq_len,
        dropout=0.0
    )

    model = GPT(config).to(device)
    model.lm_head.weight = model.wte.weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    writer = SummaryWriter("/content/logs/mini_gpt_tpu")

    xm.master_print("Starting TPU training...")

    # ---- Main training loop ----
    for step in range(max_iters):

        # Compute LR
        lr = get_lr(
            step,
            warmup_steps,
            max_iters,
            base_lr,
            min_lr=1e-5
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        xb, yb = train_loader.get_batch("train")
        xb = xb.to(device)
        yb = yb.to(device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()

        # TPU-safe optimizer step
        xm.optimizer_step(optimizer)

        # Logging (only master core prints)
        if step % 50 == 0:
            xm.master_print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", lr, step)

        # Evaluate
        if step % eval_interval == 0 and step > 0:
            with torch.no_grad():
                xb, yb = val_loader.get_batch("val")
                xb = xb.to(device)
                yb = yb.to(device)
                _, val_loss = model(xb, yb)
                xm.master_print(f"VAL LOSS: {val_loss.item():.4f}")
                writer.add_scalar("loss/val", val_loss.item(), step)

        # Save model
        if step % 2000 == 0 and step > 0:
            xm.master_print("Saving TPU checkpoint...")
            xm.save(model.state_dict(), "mini_gpt_tpu.pt")

    xm.master_print("Training complete!")
    xm.save(model.state_dict(), "mini_gpt_tpu.pt")


if __name__ == "__main__":
    train_tpu()
