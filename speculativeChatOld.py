"""
speculativeChat.py

Interactive timing tool. Type a prompt and get the answer plus timing from
up to three backends:

  baseline     the target model alone (Hugging Face, float16, KV cache)
  speculative  draft + target speculative decoding (same models, KV cache)
  ollama       the target model served by Ollama (llama.cpp)

All three use greedy decoding (temperature 0), so baseline and speculative
should produce identical text. Every query is appended to a CSV log.

Choose the model pair with MODEL_PAIR below. At the prompt, type a question,
or one of these commands:

  :mode all | baseline | speculative | ollama    choose what runs
  :k 5                                          draft length for speculative
  :max 256                                      maximum new tokens
  :help                                         show commands
  :quit                                         exit

Ollama must be running (`ollama serve`, or the Ollama app) and the model
pulled first, for example `ollama pull llama3.1:8b-instruct-fp16`.
Before the first run, log in once with `hf auth login` for the gated Llama
models. The Qwen models are not gated.
"""

import csv
import json
import os
import platform
import time
import urllib.error
import urllib.request
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Draft and target must share a tokenizer, since token ids are compared directly.
# "ollama" is the Ollama tag for the same target model. The -fp16 tags match
# the Hugging Face precision. Plain tags such as "llama3.1:8b" and "qwen2.5:7b"
# are 4-bit quantized, which is what most Ollama users run.
MODEL_PAIRS = {
    "qwen": {
        "draft": "Qwen/Qwen2.5-0.5B-Instruct",
        "target": "Qwen/Qwen2.5-7B-Instruct",
        "ollama": "qwen2.5:7b-instruct-fp16",
    },
    "llama": {
        "draft": "meta-llama/Llama-3.2-1B-Instruct",
        "target": "meta-llama/Llama-3.1-8B-Instruct",
        "ollama": "llama3.1:8b-instruct-fp16",
    },
}

MODEL_PAIR = "llama"

# What runs for each prompt when the program starts (change with :mode).
START_MODE = "all"

# Draft length for speculative decoding (change with :k).
START_K = 5

# Maximum new tokens per answer (change with :max).
START_MAX_NEW_TOKENS = 256

OLLAMA_URL = "http://localhost:11434"

# Two logs are kept. The chat log has everything, including the prompt and
# answer text. The performance log has only the measurements, one row per
# method per query; query_id links the rows that came from the same prompt.
LOG_FILE = f"chat_log_{MODEL_PAIR}.csv"
PERF_LOG_FILE = f"perf_log_{MODEL_PAIR}.csv"

MODES = ["all", "baseline", "speculative", "ollama"]

LOG_FIELDS = [
    "timestamp", "pair", "method", "model", "K", "max_new_tokens",
    "prompt", "prompt_tokens", "prompt_time", "generated_tokens", "gen_time",
    "gen_tokens_per_sec", "total_time", "load_time",
    "rounds", "accepted_tokens", "acceptance_rate", "tokens_per_round",
    "draft_time", "target_time", "matches_baseline", "answer",
]

PERF_FIELDS = [
    "timestamp", "query_id", "host", "device", "pair", "method", "model",
    "K", "max_new_tokens", "prompt_tokens", "prompt_time",
    "generated_tokens", "gen_time", "gen_tokens_per_sec", "speedup_vs_baseline",
    "total_time", "load_time",
    "rounds", "accepted_tokens", "acceptance_rate", "tokens_per_round",
    "draft_time", "target_time", "matches_baseline",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for the GPU to finish so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(name, device):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(device)
    model.eval()
    return model


def encode(tokenizer, prompt, device):
    """Format the prompt with the model's chat template."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded["input_ids"].to(device)


def get_eos_ids(tokenizer, model):
    """Instruct models can have more than one end-of-sequence token."""
    ids = set()
    gen_eos = model.generation_config.eos_token_id
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def crop_cache(cache, keep):
    """Roll the KV cache back so it holds only the first `keep` positions.
    Used to discard the entries for draft tokens the target rejected."""
    remove = cache.get_seq_length() - keep
    if remove > 0:
        cache.crop(-remove)


def ratio(a, b):
    return a / b if b else 0.0


# ------------------------------------------------------------
# Baseline: target model alone, with a KV cache
# ------------------------------------------------------------

@torch.no_grad()
def run_baseline(model, input_ids, eos_ids, vocab, max_new, device):
    """Greedy decoding, one token per forward pass.

    Prompt processing covers every prompt token except the last. Generation
    starts by feeding the last prompt token, so all generated tokens come from
    the generation loop, the same split the speculative function uses."""
    sync(device)
    start = time.perf_counter()

    cache = model(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
    sync(device)
    prompt_time = time.perf_counter() - start

    last = input_ids[0, -1].item()
    tokens = []
    while len(tokens) < max_new:
        out = model(
            input_ids=torch.tensor([[last]], device=device),
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        last = int(out.logits[0, -1, :vocab].argmax())
        tokens.append(last)
        if last in eos_ids:
            break

    sync(device)
    total_time = time.perf_counter() - start
    gen_time = total_time - prompt_time

    return tokens, {
        "prompt_tokens": input_ids.shape[1],
        "prompt_time": prompt_time,
        "generated_tokens": len(tokens),
        "gen_time": gen_time,
        "gen_tokens_per_sec": ratio(len(tokens), gen_time),
        "total_time": total_time,
    }


# ------------------------------------------------------------
# Speculative decoding, with KV caches on both models
# ------------------------------------------------------------

@torch.no_grad()
def run_speculative(draft, target, input_ids, k, eos_ids, vocab, max_new, device):
    """Greedy speculative decoding (Leviathan et al. 2023, temperature 0).

    Each round:
      1. The draft proposes k tokens, one forward pass each.
      2. The target scores all k proposals in ONE forward pass. Its k + 1
         output rows give its own choice at each proposal position plus one
         extra "bonus" position after the last proposal.
      3. Keep the longest prefix where draft and target agree, then add the
         target's own token at the first disagreement (or the bonus token if
         all k were accepted). So every round yields between 1 and k + 1 tokens.
      4. Roll both caches back to the tokens that were kept.

    Both caches always hold a prefix of `seq`. At the start of a round each
    model is fed only the tokens its cache does not yet cover."""
    seq = input_ids[0].tolist()
    new_tokens = []
    rounds = accepted_total = 0
    draft_time = target_time = 0.0

    sync(device)
    start = time.perf_counter()

    # Prompt processing: both models cache every prompt token except the last.
    d_cache = draft(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
    t_cache = target(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
    sync(device)
    prompt_time = time.perf_counter() - start

    done = False
    while not done:
        rounds += 1
        base = len(seq)

        # STEP 1: draft proposes k tokens.
        t0 = time.perf_counter()
        draft_input = seq[d_cache.get_seq_length():]
        proposed = []
        for _ in range(k):
            out = draft(
                input_ids=torch.tensor([draft_input], device=device),
                past_key_values=d_cache,
                use_cache=True,
            )
            d_cache = out.past_key_values
            token = int(out.logits[0, -1, :vocab].argmax())
            proposed.append(token)
            draft_input = [token]
        sync(device)
        draft_time += time.perf_counter() - t0

        # STEP 2: target verifies all k proposals in one forward pass.
        t0 = time.perf_counter()
        verify_input = seq[t_cache.get_seq_length():] + proposed
        out = target(
            input_ids=torch.tensor([verify_input], device=device),
            past_key_values=t_cache,
            use_cache=True,
        )
        t_cache = out.past_key_values
        target_preds = out.logits[0, -(k + 1):, :vocab].argmax(dim=-1).tolist()
        sync(device)
        target_time += time.perf_counter() - t0

        # STEP 3: accept the longest agreeing prefix, then the target's token.
        accepted = 0
        while accepted < k and proposed[accepted] == target_preds[accepted]:
            accepted += 1
        accepted_total += accepted

        for token in proposed[:accepted] + [target_preds[accepted]]:
            seq.append(token)
            new_tokens.append(token)
            if token in eos_ids or len(new_tokens) >= max_new:
                done = True
                break

        # STEP 4: roll back the cache entries for rejected tokens.
        # The target cached all k proposals; keep only the accepted ones.
        # The draft cached its first k - 1 proposals (the last was never fed
        # back in), so it keeps at most k - 1.
        crop_cache(t_cache, base + accepted)
        crop_cache(d_cache, base + min(accepted, k - 1))

    sync(device)
    total_time = time.perf_counter() - start
    gen_time = total_time - prompt_time
    n = len(new_tokens)

    return new_tokens, {
        "prompt_tokens": input_ids.shape[1],
        "prompt_time": prompt_time,
        "generated_tokens": n,
        "gen_time": gen_time,
        "gen_tokens_per_sec": ratio(n, gen_time),
        "total_time": total_time,
        "rounds": rounds,
        "accepted_tokens": accepted_total,
        "acceptance_rate": ratio(accepted_total, rounds * k),
        "tokens_per_round": ratio(n, rounds),
        "draft_time": draft_time,
        "target_time": target_time,
    }


# ------------------------------------------------------------
# Ollama
# ------------------------------------------------------------

def run_ollama(model_name, prompt, max_new):
    """Send the prompt to Ollama's chat API and read its own timing fields.
    Ollama reports durations in nanoseconds."""
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_new, "seed": 0},
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if e.code == 404:
            raise RuntimeError(f"Ollama doesn't have {model_name}. Run: ollama pull {model_name}") from e
        raise RuntimeError(f"Ollama returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_URL}. Start it with `ollama serve` or the Ollama app."
        ) from e
    wall = time.perf_counter() - start

    ns = 1e9
    gen_tokens = data.get("eval_count", 0)
    gen_time = data.get("eval_duration", 0) / ns
    return data["message"]["content"], {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "prompt_time": data.get("prompt_eval_duration", 0) / ns,
        "generated_tokens": gen_tokens,
        "gen_time": gen_time,
        "gen_tokens_per_sec": ratio(gen_tokens, gen_time),
        "total_time": wall,
        "load_time": data.get("load_duration", 0) / ns,
    }


# ------------------------------------------------------------
# Display and logging
# ------------------------------------------------------------

def print_table(results):
    header = (
        f"{'method':<13}{'prompt tok':>11}{'prompt s':>10}{'gen tok':>9}"
        f"{'gen s':>9}{'tok/s':>9}{'total s':>9}  notes"
    )
    print("\n" + header)
    print("-" * (len(header) + 20))
    for method, m in results.items():
        if method == "speculative":
            notes = (f"K={m['K']}, accept {m['acceptance_rate']:.0%}, "
                     f"{m['tokens_per_round']:.2f} tok/round")
        elif method == "ollama":
            notes = f"model load {m['load_time']:.1f}s" if m["load_time"] > 0.5 else ""
        else:
            notes = ""
        print(
            f"{method:<13}{m['prompt_tokens']:>11}{m['prompt_time']:>10.2f}"
            f"{m['generated_tokens']:>9}{m['gen_time']:>9.2f}"
            f"{m['gen_tokens_per_sec']:>9.1f}{m['total_time']:>9.2f}  {notes}"
        )

    if "baseline" in results:
        base_rate = results["baseline"]["gen_tokens_per_sec"]
        for method in ("speculative", "ollama"):
            if method in results and base_rate:
                speedup = results[method]["gen_tokens_per_sec"] / base_rate
                print(f"{method} generation speed vs baseline: {speedup:.2f}x")


def log_results(prompt, results, answers, k, max_new, names):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
        if new_file:
            writer.writeheader()
        for method, m in results.items():
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pair": MODEL_PAIR,
                "method": method,
                "model": names[method],
                "K": k if method == "speculative" else "",
                "max_new_tokens": max_new,
                "prompt": prompt,
                "answer": answers[method],
            }
            row.update({key: value for key, value in m.items() if key in LOG_FIELDS})
            writer.writerow(row)


def log_performance(query_id, results, k, max_new, names, device):
    """Measurements only, no prompt or answer text."""
    base_rate = results.get("baseline", {}).get("gen_tokens_per_sec")
    timestamp = datetime.now().isoformat(timespec="seconds")

    new_file = not os.path.exists(PERF_LOG_FILE)
    with open(PERF_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_FIELDS, restval="")
        if new_file:
            writer.writeheader()
        for method, m in results.items():
            row = {
                "timestamp": timestamp,
                "query_id": query_id,
                "host": platform.node(),
                "device": device.type,
                "pair": MODEL_PAIR,
                "method": method,
                "model": names[method],
                "K": k if method == "speculative" else "",
                "max_new_tokens": max_new,
            }
            row.update({key: value for key, value in m.items() if key in PERF_FIELDS})
            if base_rate:
                row["speedup_vs_baseline"] = m["gen_tokens_per_sec"] / base_rate
            writer.writerow(row)


def print_help():
    print(
        "\nType a prompt and press Enter. Commands:\n"
        "  :mode all | baseline | speculative | ollama\n"
        "  :k <n>      draft length for speculative decoding\n"
        "  :max <n>    maximum new tokens per answer\n"
        "  :help       show this list\n"
        "  :quit       exit"
    )


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

def main():
    pair = MODEL_PAIRS[MODEL_PAIR]
    draft_name, target_name, ollama_name = pair["draft"], pair["target"], pair["ollama"]
    names = {"baseline": target_name, "speculative": f"{draft_name} -> {target_name}",
             "ollama": ollama_name}

    device = pick_device()
    print("Using device:", device)
    print("Draft model: ", draft_name)
    print("Target model:", target_name)
    print("Ollama model:", ollama_name)

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    if tokenizer.get_vocab() != AutoTokenizer.from_pretrained(draft_name).get_vocab():
        raise SystemExit("Draft and target tokenizers differ. Pick models that share a vocabulary.")

    # Output layers can be padded wider than the real vocabulary (Qwen2.5 0.5B
    # and 7B are padded to different widths), so logits are cut to this size.
    vocab = len(tokenizer)

    print("Loading draft model...")
    draft = load_model(draft_name, device)
    print("Loading target model...")
    target = load_model(target_name, device)
    eos_ids = get_eos_ids(tokenizer, target)

    mode, k, max_new = START_MODE, START_K, START_MAX_NEW_TOKENS

    # Warmup so GPU start-up costs are not counted in the first real query.
    print("Warming up...")
    warm_ids = encode(tokenizer, "Say hello.", device)
    run_baseline(target, warm_ids, eos_ids, vocab, 8, device)
    run_speculative(draft, target, warm_ids, k, eos_ids, vocab, 8, device)

    print(f"\nReady. Pair: {MODEL_PAIR}. Results are appended to {LOG_FILE}.")
    print_help()

    while True:
        try:
            line = input(f"\n[{mode}, K={k}, max={max_new}] prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        # Commands
        if line.startswith(":"):
            parts = line[1:].split()
            command = parts[0].lower() if parts else ""
            if command in ("quit", "q", "exit"):
                break
            elif command == "help":
                print_help()
            elif command == "mode" and len(parts) == 2 and parts[1] in MODES:
                mode = parts[1]
            elif command == "k" and len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 1:
                k = int(parts[1])
            elif command == "max" and len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 1:
                max_new = int(parts[1])
            else:
                print("Unrecognized command.")
                print_help()
            continue

        # A prompt
        prompt = line
        methods = ["baseline", "speculative", "ollama"] if mode == "all" else [mode]
        results, answers = {}, {}
        baseline_tokens = None

        input_ids = encode(tokenizer, prompt, device)

        for method in methods:
            print(f"Running {method}...", flush=True)
            if method == "baseline":
                tokens, metrics = run_baseline(target, input_ids, eos_ids, vocab, max_new, device)
                baseline_tokens = tokens
                answers[method] = tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            elif method == "speculative":
                tokens, metrics = run_speculative(
                    draft, target, input_ids, k, eos_ids, vocab, max_new, device
                )
                metrics["K"] = k
                if baseline_tokens is not None:
                    metrics["matches_baseline"] = tokens == baseline_tokens
                answers[method] = tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            else:
                try:
                    answers[method], metrics = run_ollama(ollama_name, prompt, max_new)
                except RuntimeError as e:
                    print(f"  Skipped: {e}")
                    continue
            results[method] = metrics

        # Answers. Baseline and speculative should match exactly, so the
        # speculative text is only printed when there is no baseline to show.
        for method in ("baseline", "speculative", "ollama"):
            if method not in answers:
                continue
            if method == "speculative" and "baseline" in answers:
                same = results["speculative"].get("matches_baseline")
                print("\n=== speculative: " + ("identical to baseline" if same else "DIFFERS from baseline:"))
                if not same:
                    print(answers[method])
                continue
            print(f"\n=== {method} ===")
            print(answers[method])

        if results:
            print_table(results)
            log_results(prompt, results, answers, k, max_new, names)

    print("Goodbye.")


if __name__ == "__main__":
    main()
