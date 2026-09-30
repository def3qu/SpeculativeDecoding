"""
speculative.py

Greedy speculative decoding vs. a plain greedy baseline on the same target
model, with the measurements needed to test the analytic model from
Leviathan et al. (2023):

    expected tokens per iteration   E = (1 - a^(K+1)) / (1 - a)
    predicted speedup               S = E / (c*K + 1)

a (alpha) is the per-token acceptance rate; c is the cost of one draft
forward pass relative to one target forward pass.

Run directly for a single configuration, or import it from sweep.py.

Authentication: set HF_TOKEN in your environment (or run
`huggingface-cli login` once) so the gated Llama weights can download.
"""

import statistics
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# DRAFT_MODEL = "HuggingFaceTB/SmolLM-135M"
# TARGET_MODEL = "HuggingFaceTB/SmolLM-1.7B"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

PROMPT = "Explain how a binary search tree works."
MAX_NEW_TOKENS = 100
K = 5            # tokens proposed by the draft model per iteration
TRIALS = 3       # timed runs per method, after one warmup
DTYPE = torch.float16

# ------------------------------------------------------------
# Device and timing helpers
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def sync():
    """GPU work is asynchronous; wait for it before reading the clock."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def now():
    sync()
    return time.perf_counter()


# ------------------------------------------------------------
# Load tokenizer and models (once, at import)
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)


def load(name):
    print("Loading", name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to(device)
    model.eval()
    return model


draft_model = load(DRAFT_MODEL)
target_model = load(TARGET_MODEL)

eos = target_model.generation_config.eos_token_id
EOS_IDS = set(eos if isinstance(eos, list) else [eos])


def encode(prompt):
    """Instruct models expect the chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"].to(device)


@torch.no_grad()
def step(model, cache, seq):
    """Feed every token of seq the cache has not seen yet; return their logits."""
    unseen = seq[:, cache.get_seq_length():]
    return model(input_ids=unseen, past_key_values=cache, use_cache=True).logits


# ------------------------------------------------------------
# Baseline: target model alone, greedy, with KV cache
# ------------------------------------------------------------

@torch.no_grad()
def run_baseline(prompt_ids, max_new):
    cache = DynamicCache()
    seq = prompt_ids
    step_times = []

    start = now()
    for _ in range(max_new):
        t0 = now()
        token = step(target_model, cache, seq)[:, -1].argmax(-1, keepdim=True)
        step_times.append(now() - t0)
        seq = torch.cat([seq, token], dim=1)
        if token.item() in EOS_IDS:
            break
    elapsed = now() - start

    return {
        "ids": seq[0, prompt_ids.shape[1]:].tolist(),
        "time": elapsed,
        "step_times": step_times[1:],   # first step is the prompt prefill
    }


# ------------------------------------------------------------
# Speculative decoding: greedy verification, with KV caches
# ------------------------------------------------------------

@torch.no_grad()
def run_speculative(prompt_ids, max_new, k):
    draft_cache, target_cache = DynamicCache(), DynamicCache()
    seq = prompt_ids
    n_new = 0
    accepted_counts, draft_step_times, verify_times = [], [], []

    start = now()

    # Prefill both caches with all but the last prompt token.
    step(draft_model, draft_cache, seq[:, :-1])
    step(target_model, target_cache, seq[:, :-1])

    done = False
    while not done:
        # STEP 1: draft proposes k tokens, one at a time.
        t0 = now()
        draft_seq = seq
        for _ in range(k):
            token = step(draft_model, draft_cache, draft_seq)[:, -1].argmax(-1, keepdim=True)
            draft_seq = torch.cat([draft_seq, token], dim=1)
        draft_step_times.append((now() - t0) / k)
        drafts = draft_seq[0, seq.shape[1]:]

        # STEP 2: target scores the last accepted token plus all k drafts
        # in ONE forward pass, giving k+1 predictions.
        t0 = now()
        logits = step(target_model, target_cache, draft_seq)
        target_pred = logits[0, -(k + 1):].argmax(-1)
        verify_times.append(now() - t0)

        # STEP 3: accept the longest prefix where draft and target agree.
        matches = (drafts == target_pred[:k]).int()
        a = int(matches.cumprod(0).sum())
        accepted_counts.append(a)

        # STEP 4: keep the accepted drafts plus one target token. If a < k it
        # is the correction; if a == k it is a free "bonus" token.
        new = torch.cat([drafts[:a], target_pred[a:a + 1]]).tolist()
        for j, t in enumerate(new):
            if t in EOS_IDS:
                new, done = new[:j + 1], True
                break
        if len(new) >= max_new - n_new:
            new, done = new[:max_new - n_new], True

        seq = torch.cat([seq, torch.tensor([new], device=device)], dim=1)
        n_new += len(new)

        # Roll back cache entries for rejected drafts. Caches cover every
        # token except the newest one, which is fed on the next step.
        keep = seq.shape[1] - 1
        target_cache.crop(keep)
        draft_cache.crop(keep)

    elapsed = now() - start

    return {
        "ids": seq[0, prompt_ids.shape[1]:].tolist(),
        "time": elapsed,
        "accepted": accepted_counts,
        "draft_step_times": draft_step_times,
        "verify_times": verify_times,
    }


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def warmup():
    """The first MPS run includes kernel compilation."""
    ids = encode("Hello")
    run_baseline(ids, 10)
    run_speculative(ids, 10, K)


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(base, spec, k):
    """Turn lists of baseline and speculative runs into one row of metrics."""
    accepted = [a for r in spec for a in r["accepted"]]
    iterations = len(accepted)
    evaluated = sum(a + 1 if a < k else k for a in accepted)
    alpha = sum(accepted) / evaluated

    t_target = statistics.mean(t for r in base for t in r["step_times"])
    t_draft = statistics.mean(t for r in spec for t in r["draft_step_times"])
    t_verify = statistics.mean(t for r in spec for t in r["verify_times"])
    c = t_draft / t_target

    base_tps = statistics.mean(len(r["ids"]) / r["time"] for r in base)
    spec_tps = statistics.mean(len(r["ids"]) / r["time"] for r in spec)

    E = (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1
    counts = Counter(accepted)
    diff = first_difference(base[0]["ids"], spec[0]["ids"])

    return {
        "k": k,
        "baseline_tps": round(base_tps, 3),
        "spec_tps": round(spec_tps, 3),
        "measured_speedup": round(spec_tps / base_tps, 3),
        "predicted_speedup": round(E / (c * k + 1), 3),
        "alpha": round(alpha, 4),
        "c": round(c, 4),
        "verify_ratio": round(t_verify / t_target, 3),
        "tokens_per_iter_measured": round(sum(len(r["ids"]) for r in spec) / iterations, 3),
        "tokens_per_iter_predicted": round(E, 3),
        "iterations": iterations,
        "outputs_match": diff is None,
        "first_diff": "" if diff is None else diff,
        "accept_hist": ";".join(f"{j}:{counts[j]}" for j in range(k + 1)),
    }


def print_report(m, text):
    print("\nOutputs identical to baseline:",
          "yes" if m["outputs_match"] else f"NO (first difference at token {m['first_diff']})")
    print("\n" + text)
    print("\n" + "=" * 60)
    print(f"Baseline tokens/sec:        {m['baseline_tps']:8.2f}")
    print(f"Speculative tokens/sec:     {m['spec_tps']:8.2f}")
    print(f"Measured speedup:           {m['measured_speedup']:8.2f}x")
    print(f"Predicted speedup:          {m['predicted_speedup']:8.2f}x")
    print("-" * 60)
    print(f"alpha (per-token accept):   {m['alpha']:8.3f}")
    print(f"c (draft / target step):    {m['c']:8.3f}")
    print(f"Verify pass / target step:  {m['verify_ratio']:8.2f}x  "
          f"(near 1 means verification is nearly free)")
    print(f"Tokens per iteration:       {m['tokens_per_iter_measured']:8.2f} measured, "
          f"{m['tokens_per_iter_predicted']:.2f} predicted")

    # Accepted-count distribution vs. the truncated geometric the theory assumes
    alpha, k, n = m["alpha"], m["k"], m["iterations"]
    counts = dict(pair.split(":") for pair in m["accept_hist"].split(";"))
    print("\nAccepted per iteration   observed   predicted")
    for j in range(k + 1):
        p = alpha ** j * (1 - alpha) if j < k else alpha ** k
        print(f"  {j:2d}                     {int(counts[str(j)]) / n:6.1%}     {p:6.1%}")


def main():
    print("Device:", device)
    print("Draft:", DRAFT_MODEL, "| Target:", TARGET_MODEL, "| K =", K)
    warmup()

    ids = encode(PROMPT)
    base = [run_baseline(ids, MAX_NEW_TOKENS) for _ in range(TRIALS)]
    spec = [run_speculative(ids, MAX_NEW_TOKENS, K) for _ in range(TRIALS)]

    metrics = summarize(base, spec, K)
    print_report(metrics, tokenizer.decode(spec[0]["ids"], skip_special_tokens=True))


if __name__ == "__main__":
    main()
