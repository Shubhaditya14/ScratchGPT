import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time

# IMPORTS FOR YOUR PROJECT STRUCTURE
from model.gpt import GPT, GPTConfig
from data.loader import DataLoader
from tokenizer import encode


# LR scheduler with warmup + cosine decay
def get_lr(step, warmup_steps, max_steps, base_lr, min_lr=0.0):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)

    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
    return min_lr + (base_lr - min_lr) * cosine


# -----------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded dataset with {len(text)} characters.")
    return text


# -----------------------------------------------------
# 2. Tokenize dataset
# -----------------------------------------------------
def tokenize_text(text):
    print("Tokenizing...")
    tokens = encode(text)
    print(f"Tokenized into {len(tokens)} tokens.")
    return tokens


# -----------------------------------------------------
# 3. Training script
# -----------------------------------------------------
def train():

    # Hyperparameters
    batch_size = 8
    seq_len = 128
    base_lr = 3e-4
    max_iters = 2000
    warmup_steps = 400
    eval_interval = 200

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    text = load_dataset("data/shakespeare.txt")
    tokens = tokenize_text(text)

    # Split dataset (90% train, 10% val)
    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens)} | Val tokens: {len(val_tokens)}")

    # Create dataloaders
    train_loader = DataLoader(train_tokens, batch_size, seq_len)
    val_loader = DataLoader(val_tokens, batch_size, seq_len)

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------
    config = GPTConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=4,
        seq_len=seq_len,
        dropout=0.0
    )

    model = GPT(config).to(device)

    # Weight tying
    model.lm_head.weight = model.wte.weight

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    writer = SummaryWriter(log_dir="runs/mini_gpt")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for step in range(max_iters):

        # Update LR
        lr = get_lr(step, warmup_steps, max_iters, base_lr, min_lr=1e-5)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        xb, yb = train_loader.get_batch(split="train")
        xb = xb.to(device)
        yb = yb.to(device)

        # Forward
        logits, loss = model(xb, yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", lr, step)

        # Eval
        if step % eval_interval == 0 and step > 0:
            with torch.no_grad():
                xb, yb = val_loader.get_batch(split="val")
                xb = xb.to(device)
                yb = yb.to(device)
                _, val_loss = model(xb, yb)
                print(f"Validation loss: {val_loss.item():.4f}")
                writer.add_scalar("loss/val", val_loss.item(), step)

    # -------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/mini_gpt.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
