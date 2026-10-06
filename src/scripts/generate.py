import torch
import torch.nn.functional as F

from src.models.gpt import GPT, GPTConfig
from src.tokenizer import encode, decode


# -----------------------------------------------------
# Top-k filtering helper
# -----------------------------------------------------
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, -1].unsqueeze(1)] = -float("Inf")
    return out


# -----------------------------------------------------
# Generate function
# -----------------------------------------------------
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    """
    Generate tokens from prompt.

    Args:
        model: GPT model
        idx: (B, T) input token indices
        max_new_tokens: number of tokens to generate
        temperature: 0.8 is sweet spot (higher=more random, <1=more deterministic)
        top_k: only sample from top-k most likely tokens (40 is typical)
    """
    model.eval()

    for _ in range(max_new_tokens):
        # idx: (B, T)
        idx_cond = idx[:, -model.config.seq_len :]

        # forward pass
        logits = model(idx_cond)  # (B, T, vocab_size)

        # take final token's logits and apply temperature
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # top-k filtering (remove low probability tokens)
        if top_k is not None:
            # Make sure top_k doesn't exceed vocab size
            k = min(top_k, logits.size(-1))
            logits = top_k_logits(logits, k)

        # convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from distribution
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

    return idx


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    # Same config as training
    config = GPTConfig(
        vocab_size=50257, n_embd=128, n_head=4, n_layer=4, seq_len=128, dropout=0.0
    )

    model = GPT(config)
    model.lm_head.weight = model.wte.weight  # weight tying (must repeat)

    # LOAD YOUR CHECKPOINT
    ckpt_path = "checkpoints/mini_gpt.pt"
    print(f"Loading checkpoint from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.to(device)
    model.eval()

    # STARTING PROMPT
    prompt = "Once upon a time"
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    # GENERATE TOKENS
    out = generate(model, idx, max_new_tokens=200, temperature=0.8, top_k=40)

    # DECODE TO TEXT
    text = decode(out[0].tolist())
    print("\n=== GENERATED TEXT ===\n")
    print(text)
    print("\n=======================\n")


if __name__ == "__main__":
    main()
