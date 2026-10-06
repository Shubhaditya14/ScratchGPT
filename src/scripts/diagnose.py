#!/usr/bin/env python3
"""
Diagnostic script to check if your model is properly trained.
Run this before worrying about generation quality.
"""

import os
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from src.models.gpt import GPT, GPTConfig
from src.data.loader import DataLoader
from src.tokenizer import encode


def diagnose():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\n")

    # ---- Check checkpoint exists ----
    ckpt_path = "checkpoints/mini_gpt.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("Run: python -m src.scripts.train")
        return

    # ---- Load checkpoint ----
    print(f"✓ Checkpoint found: {ckpt_path}")
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  Size: {ckpt_size_mb:.1f} MB\n")

    # ---- Load model ----
    config = GPTConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=4,
        seq_len=128,
        dropout=0.0,
    )

    model = GPT(config)
    model.lm_head.weight = model.wte.weight
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded")
    print(f"  Total parameters: {total_params:,}\n")

    # ---- Evaluate on validation set ----
    print("Evaluating on validation set...")
    text = open("src/data/shakespeare.txt").read()
    tokens = encode(text)
    split_idx = int(0.9 * len(tokens))
    val_tokens = tokens[split_idx:]

    val_loader = DataLoader(val_tokens, batch_size=8, seq_len=128)

    total_loss = 0
    steps = 50

    with torch.no_grad():
        for _ in range(steps):
            xb, yb = val_loader.get_batch(split="val")
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()

    avg_loss = total_loss / steps
    print(f"✓ Validation loss: {avg_loss:.4f}")

    # ---- Diagnosis ----
    print("\n" + "=" * 50)
    print("DIAGNOSIS")
    print("=" * 50)

    if avg_loss > 3.5:
        print(f"⚠️  Loss is HIGH ({avg_loss:.2f}). Model may not be trained enough.")
        print("   → Run more training iterations")
        print("   → Check that training loss is decreasing")
    elif avg_loss > 2.5:
        print(f"⚠️  Loss is MODERATE ({avg_loss:.2f}). Model has basic training.")
        print("   → Should produce somewhat coherent output")
    else:
        print(f"✓ Loss is GOOD ({avg_loss:.2f}). Model is well-trained!")
        print("   → Should produce coherent Shakespeare")

    # ---- Test sampling ----
    print("\n" + "=" * 50)
    print("SAMPLING TEST")
    print("=" * 50)

    prompt = "ROMEO:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {encode(prompt)}\n")

    # Generate with default settings
    print("Generating 50 tokens (temp=0.8, top_k=40)...")
    from src.scripts.generate import generate

    with torch.no_grad():
        out = generate(model, idx, max_new_tokens=50, temperature=0.8, top_k=40)
        from src.tokenizer import decode

        text = decode(out[0].tolist())
        print(f"\nGenerated:\n{text}\n")

    # Check for repetition collapse
    gen_tokens = out[0, len(idx[0]) :].tolist()
    unique_tokens = len(set(gen_tokens))
    unique_ratio = unique_tokens / len(gen_tokens)

    print(f"Generated {len(gen_tokens)} tokens")
    print(f"Unique tokens: {unique_tokens} ({unique_ratio * 100:.1f}%)")

    if unique_ratio < 0.3:
        print("⚠️  LOW DIVERSITY - possible repetition collapse")
        print("   → Lower temperature (try 0.6)")
        print("   → Increase top_k (try 50 or 60)")
    elif unique_ratio < 0.6:
        print("⚠️  MODERATE DIVERSITY - model might need more training")
    else:
        print("✓ GOOD DIVERSITY - sampling is working")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    diagnose()
