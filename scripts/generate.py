import torch
import torch.nn.functional as F

from model.gpt import GPT, GPTConfig
from tokenizer import encode, decode


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
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()

    for _ in range(max_new_tokens):

        # idx: (B, T)
        idx_cond = idx[:, -model.config.seq_len:]

        # forward
        logits = model(idx_cond)

        # take final token's logits
        logits = logits[:, -1, :] / temperature

        # top-k
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)

        # sample
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

    return idx


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # Same config as training
    config = GPTConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=4,
        seq_len=128,
        dropout=0.0
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
    out = generate(
        model,
        idx,
        max_new_tokens=200,
        temperature=1.0,
        top_k=50
    )

    # DECODE TO TEXT
    text = decode(out[0].tolist())
    print("\n=== GENERATED TEXT ===\n")
    print(text)
    print("\n=======================\n")


if __name__ == "__main__":
    main()
