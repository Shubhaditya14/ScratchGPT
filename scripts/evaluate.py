import torch
from model.gpt import GPT, GPTConfig
from data.loader import DataLoader
from tokenizer import encode


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_text(text):
    return encode(text)


def main():

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # ---- Load dataset ----
    text = load_dataset("data/shakespeare.txt")
    tokens = tokenize_text(text)

    # val = last 10%
    split_idx = int(0.9 * len(tokens))
    val_tokens = tokens[split_idx:]

    val_loader = DataLoader(val_tokens, batch_size=8, seq_len=128)

    # ---- Load model ----
    config = GPTConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=4,
        seq_len=128,
        dropout=0.0
    )

    model = GPT(config).to(device)
    model.lm_head.weight = model.wte.weight

    ckpt_path = "checkpoints/mini_gpt.pt"
    print("Loading checkpoint:", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()

    # ---- Evaluate ----
    total_loss = 0
    steps = 100  # evaluate on 100 batches

    with torch.no_grad():
        for _ in range(steps):
            xb, yb = val_loader.get_batch(split="val")
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()

    avg_loss = total_loss / steps
    print(f"\nValidation Loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    main()
