"""
show_tokens.py

Visual aid: load a small model, ask for a prompt, and show the first k
tokens it generates, one at a time, with each token's id, the model's
confidence, and the runner-up tokens it considered.

This is exactly what the draft model does in speculative decoding: greedily
pick the most likely next token, append it, and repeat k times.

Usage:
    python show_tokens.py        # k = 5
    python show_tokens.py 8      # k = 8
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TOP = 3          # how many candidate tokens to show per position

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float32 if device.type == "cpu" else torch.float16

print(f"Loading {MODEL} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)
model.eval()

eos = model.generation_config.eos_token_id
EOS_IDS = set(eos if isinstance(eos, list) else [eos])


def show(tok_id):
    """Token text with visible spaces and newlines."""
    return tokenizer.decode([tok_id]).replace(" ", "␣").replace("\n", "↵")


def encode(prompt):
    """Chat template for Instruct models; plain text for base models."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].to(device)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


# ------------------------------------------------------------
# Generate k tokens, recording the top candidates at each step
# ------------------------------------------------------------

@torch.no_grad()
def first_k_tokens(prompt, k):
    ids = encode(prompt)
    cache = DynamicCache()
    next_input = ids                    # first pass reads the whole prompt
    rows = []

    for _ in range(k):
        logits = model(input_ids=next_input, past_key_values=cache, use_cache=True).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, TOP)

        chosen = top.indices[0].item()  # greedy: the most likely token
        rows.append((top.indices.tolist(), top.values.tolist()))

        next_input = torch.tensor([[chosen]], device=device)   # later passes read one token
        if chosen in EOS_IDS:
            break

    return ids.shape[1], rows


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

def display(prompt, n_prompt_tokens, rows):
    pieces = [show(t) for t in tokenizer.encode(prompt, add_special_tokens=False)]
    print(f"\nYour prompt as tokens ({len(pieces)}): " + " | ".join(pieces))
    print(f"Model input with chat formatting: {n_prompt_tokens} tokens")

    print(f"\n{'#':>2}  {'token':<14}{'id':>7}  {'confidence':<27}  runners-up")
    print("-" * 78)
    for i, (idxs, probs) in enumerate(rows, 1):
        bar = "█" * round(probs[0] * 20)
        alts = ", ".join(f"{show(t)!r} {p:.0%}" for t, p in zip(idxs[1:], probs[1:]))
        print(f"{i:>2}  {show(idxs[0])!r:<14}{idxs[0]:>7}  {probs[0]:>5.0%} {bar:<20}  {alts}")

    chosen = [idxs[0] for idxs, _ in rows]
    print(f"\nTogether: {tokenizer.decode(chosen)!r}")


def main():
    print(f"\nShowing the first {K} tokens. Press Enter on a blank line to quit.")
    while True:
        prompt = input("\nPrompt: ").strip()
        if not prompt:
            break
        n_prompt, rows = first_k_tokens(prompt, K)
        display(prompt, n_prompt, rows)


if __name__ == "__main__":
    main()
