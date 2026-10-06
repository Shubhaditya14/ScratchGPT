#!/usr/bin/env python3
import curses
import os
import sys
import textwrap
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from src.models.gpt import GPT, GPTConfig
from src.tokenizer import encode, decode


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, -1].unsqueeze(1)] = -float("Inf")
    return out


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    """
    Generate tokens from prompt.

    Args:
        temperature: 0.8 is sweet spot (higher=more random, <1=more deterministic)
        top_k: only sample from top-k most likely tokens (40 is typical)
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.seq_len :]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            logits = top_k_logits(logits, k)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def wrap_lines(text, width):
    if width <= 1:
        return [text]
    lines = []
    for raw in text.splitlines() or [""]:
        wrapped = textwrap.wrap(
            raw,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped if wrapped else [""])
    return lines


def render_output(win, lines):
    win.erase()
    h, w = win.getmaxyx()
    start = max(0, len(lines) - h)
    visible = lines[start:]
    for i, line in enumerate(visible):
        win.addnstr(i, 0, line, w - 1)
    win.noutrefresh()


def render_status(win, status):
    win.erase()
    h, w = win.getmaxyx()
    win.addnstr(0, 0, status.ljust(w - 1), w - 1)
    win.noutrefresh()


def input_line(win, prompt):
    win.erase()
    win.addstr(0, 0, prompt)
    win.refresh()
    buf = []
    while True:
        ch = win.getch()
        if ch in (10, 13):
            break
        if ch in (3,):
            raise KeyboardInterrupt
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if buf:
                buf.pop()
        elif 32 <= ch <= 126:
            buf.append(chr(ch))
        win.erase()
        win.addstr(0, 0, prompt)
        win.addnstr(
            0, len(prompt), "".join(buf), max(1, win.getmaxyx()[1] - len(prompt) - 1)
        )
        win.refresh()
    return "".join(buf).strip()


def load_model(device, ckpt_path):
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
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def tui(stdscr):
    curses.curs_set(1)
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = str(PROJECT_DIR / "checkpoints" / "mini_gpt.pt")
    temperature = 0.8
    top_k = 40
    max_new_tokens = 200

    output_lines = ["ScratchGPT TUI", "Type /help for commands."]

    h, w = stdscr.getmaxyx()
    output_win = curses.newwin(h - 2, w, 0, 0)
    input_win = curses.newwin(1, w, h - 2, 0)
    status_win = curses.newwin(1, w, h - 1, 0)

    model = None
    status = f"Loading model on {device}..."
    render_status(status_win, status)
    render_output(output_win, output_lines)
    curses.doupdate()

    try:
        if os.path.exists(ckpt_path):
            model = load_model(device, ckpt_path)
            status = f"Ready | device={device} | ckpt={ckpt_path}"
        else:
            status = f"Checkpoint not found: {ckpt_path}"
            output_lines.append(status)
    except Exception as exc:
        status = f"Failed to load model: {exc}"
        output_lines.append(status)

    while True:
        render_output(output_win, output_lines)
        render_status(
            status_win,
            f"temp={temperature} top_k={top_k} max={max_new_tokens} | {status}",
        )
        curses.doupdate()

        try:
            user_input = input_line(input_win, "> ")
        except KeyboardInterrupt:
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("/quit", "/exit"):
                break
            if cmd == "/clear":
                output_lines = ["ScratchGPT TUI", "Type /help for commands."]
                continue
            if cmd == "/help":
                output_lines.extend(
                    wrap_lines(
                        "Commands: /help /quit /clear /temp <float> /topk <int|none> /max <int> /ckpt <path>",
                        w - 1,
                    )
                )
                continue
            if cmd == "/temp" and args:
                try:
                    temperature = float(args[0])
                    status = "Updated temperature"
                except ValueError:
                    status = "Invalid temperature"
                continue
            if cmd == "/topk" and args:
                val = args[0].lower()
                if val in ("none", "0"):
                    top_k = None
                    status = "Disabled top_k"
                else:
                    try:
                        top_k = int(val)
                        status = "Updated top_k"
                    except ValueError:
                        status = "Invalid top_k"
                continue
            if cmd == "/max" and args:
                try:
                    max_new_tokens = int(args[0])
                    status = "Updated max tokens"
                except ValueError:
                    status = "Invalid max tokens"
                continue
            if cmd == "/ckpt" and args:
                ckpt_path = args[0]
                status = "Loading checkpoint..."
                render_status(status_win, status)
                curses.doupdate()
                try:
                    model = load_model(device, ckpt_path)
                    status = f"Loaded {ckpt_path}"
                except Exception as exc:
                    status = f"Failed to load checkpoint: {exc}"
                continue

            status = "Unknown command"
            continue

        output_lines.extend(wrap_lines(f"You: {user_input}", w - 1))
        if model is None:
            output_lines.extend(wrap_lines("AI: model not loaded", w - 1))
            status = "No model loaded"
            continue

        status = "Generating..."
        render_status(status_win, status)
        render_output(output_win, output_lines)
        curses.doupdate()

        try:
            idx = torch.tensor([encode(user_input)], dtype=torch.long).to(device)
            out = generate(
                model,
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            out_tokens = out[0].tolist()
            gen_tokens = out_tokens[len(idx[0]) :]
            response = decode(gen_tokens)
        except Exception as exc:
            response = f"Error: {exc}"

        output_lines.extend(wrap_lines(f"AI: {response}", w - 1))
        status = "Ready"


def main():
    curses.wrapper(tui)


if __name__ == "__main__":
    main()
