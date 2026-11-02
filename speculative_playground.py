#!/usr/bin/env python3
"""
speculative_playground.py

An interactive program for trying speculative decoding on your own computer.

Type a prompt and the program generates the answer two ways:

  baseline     the large "target" model alone, one token per forward pass
  speculative  a small "draft" model guesses K tokens, then the target checks
               all K guesses in a single forward pass

Both use greedy decoding (always pick the most likely token), so the two
answers should be identical. Only the speed differs.

Things to try:

  * Compare speeds:             just type a prompt (default mode is "compare")
  * Watch the guesses:          :show on    (green = accepted, red = rejected)
  * Change the draft length:    :k 3        (try 1, 3, 5, 8)
  * Find the best K:            :sweep      (runs K = 1, 2, 3, 5, 8 and charts them)
  * See how the task matters:   :examples   (math, translation, code, and more)

Setup (Python 3.9 or newer):

    pip install torch transformers

The default model pair (Qwen2.5 0.5B drafting for Qwen2.5 3B) needs about
8 GB of free memory and does not require a Hugging Face account. Smaller and
larger pairs are available with --pair (see --help). The first run downloads
the models, which can take a few minutes.

Run it:

    python speculative_playground.py
    python speculative_playground.py --pair tiny
    python speculative_playground.py --prompt "Explain how a hash table works."
"""

import argparse
import os
import sys
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("This program needs PyTorch and Transformers. Install them with:\n"
             "    pip install torch transformers")


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Draft and target must share a tokenizer, because their token ids are
# compared directly. All Qwen2.5 models share one tokenizer, and all
# Llama 3.x models share another.
PAIRS = {
    "tiny": {
        "draft": "Qwen/Qwen2.5-0.5B-Instruct",
        "target": "Qwen/Qwen2.5-1.5B-Instruct",
        "note": "about 4 GB of memory; good for older laptops",
    },
    "small": {
        "draft": "Qwen/Qwen2.5-0.5B-Instruct",
        "target": "Qwen/Qwen2.5-3B-Instruct",
        "note": "about 8 GB of memory (default)",
    },
    "qwen": {
        "draft": "Qwen/Qwen2.5-0.5B-Instruct",
        "target": "Qwen/Qwen2.5-7B-Instruct",
        "note": "about 16 GB of memory",
    },
    "llama": {
        "draft": "meta-llama/Llama-3.2-1B-Instruct",
        "target": "meta-llama/Llama-3.1-8B-Instruct",
        "note": "about 18 GB of memory; gated, run `hf auth login` first",
    },
}
DEFAULT_PAIR = "small"

DEFAULT_K = 5
DEFAULT_MAX_NEW = 128
SWEEP_KS = [1, 2, 3, 5, 8]
TRACE_ROUNDS_SHOWN = 15

EXAMPLES = [
    ("facts", "Explain how a binary search tree works."),
    ("code", "Write a Python function that checks whether a string is a palindrome."),
    ("math", "A train leaves at 3:15 pm and travels 210 miles at 60 miles per hour. "
             "When does it arrive? Show your steps."),
    ("translation", "Translate into French: The library opens at nine, and the reading "
                    "room stays open until midnight."),
    ("summary", "Summarize the following text in two sentences:\n\n"
                "Honeybees communicate the location of food through a behavior called the "
                "waggle dance. A forager returning to the hive moves in a figure-eight "
                "pattern, and the angle of the straight middle run relative to vertical "
                "indicates the direction of the food relative to the sun. The duration of "
                "the waggle indicates the distance. Other bees follow the dancer and then "
                "fly out to find the source themselves."),
    ("creative", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
]

MODES = ["compare", "baseline", "speculative"]


# ------------------------------------------------------------
# Terminal colors (turned off automatically when not supported)
# ------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if os.name == "nt":
    os.system("")  # lets Windows terminals understand color codes


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(requested, device):
    if requested != "auto":
        return getattr(torch, requested)
    return torch.float32 if device.type == "cpu" else torch.float16


def sync(device):
    """Wait for the GPU to finish so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(name, device, dtype):
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # older versions of Transformers call it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model


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
    Used to throw away the entries for draft tokens the target rejected."""
    remove = cache.get_seq_length() - keep
    if remove > 0:
        cache.crop(-remove)


def ratio(a, b):
    return a / b if b else 0.0


def predicted_speedup(alpha, k, c):
    """Speedup formula from Leviathan et al. (2023):
         (1 - alpha^(K+1)) / ((1 - alpha)(K*c + 1))
    alpha is the chance the target accepts a draft token, and c is the cost
    of one draft pass relative to one target pass."""
    if alpha is None or c is None:
        return None
    if alpha >= 1.0:
        return (k + 1) / (k * c + 1)
    return (1 - alpha ** (k + 1)) / ((1 - alpha) * (k * c + 1))


# ------------------------------------------------------------
# Baseline: the target model alone
# ------------------------------------------------------------

@torch.no_grad()
def run_baseline(model, input_ids, eos_ids, vocab, max_new, device):
    """Greedy decoding with a KV cache, one token per forward pass."""
    sync(device)
    start = time.perf_counter()

    # Process every prompt token except the last. Generation then starts by
    # feeding the last prompt token, so every new token comes from the loop.
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
    total = time.perf_counter() - start
    gen_time = total - prompt_time
    return tokens, {
        "tokens": len(tokens),
        "gen_time": gen_time,
        "tok_per_sec": ratio(len(tokens), gen_time),
        "prompt_time": prompt_time,
    }


# ------------------------------------------------------------
# Speculative decoding
# ------------------------------------------------------------

@torch.no_grad()
def run_speculative(draft, target, input_ids, k, eos_ids, vocab, max_new, device):
    """Greedy speculative decoding (Leviathan et al., 2023, temperature 0).

    Each round:
      1. The draft model proposes k tokens, one forward pass each.
      2. The target scores all k proposals in ONE forward pass. Its k + 1
         output rows give its own choice at each proposal position, plus one
         extra "bonus" position after the last proposal.
      3. Keep the longest prefix where draft and target agree, then add the
         target's own token at the first disagreement (or the bonus token if
         all k were accepted). Every round yields between 1 and k + 1 tokens.
      4. Roll both KV caches back to the tokens that were kept.
    """
    seq = input_ids[0].tolist()
    new_tokens = []
    trace = []  # one entry per round: (proposed tokens, number accepted, target's token)
    rounds = accepted_total = 0
    draft_time = target_time = 0.0

    sync(device)
    start = time.perf_counter()

    d_cache = draft(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
    t_cache = target(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
    sync(device)
    prompt_time = time.perf_counter() - start

    done = False
    while not done:
        rounds += 1
        base = len(seq)

        # Step 1: the draft model proposes k tokens.
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

        # Step 2: the target checks all k proposals in one forward pass.
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

        # Step 3: accept the longest agreeing prefix, then the target's token.
        accepted = 0
        while accepted < k and proposed[accepted] == target_preds[accepted]:
            accepted += 1
        accepted_total += accepted
        trace.append((proposed, accepted, target_preds[accepted]))

        for token in proposed[:accepted] + [target_preds[accepted]]:
            seq.append(token)
            new_tokens.append(token)
            if token in eos_ids or len(new_tokens) >= max_new:
                done = True
                break

        # Step 4: roll back the cache entries for rejected tokens.
        # The target cached all k proposals; keep only the accepted ones.
        # The draft cached its first k - 1 proposals (the last was never fed
        # back in), so it keeps at most k - 1.
        crop_cache(t_cache, base + accepted)
        crop_cache(d_cache, base + min(accepted, k - 1))

    sync(device)
    total = time.perf_counter() - start
    gen_time = total - prompt_time
    n = len(new_tokens)

    # Each round tests the accepted tokens plus the first rejected one, so the
    # fair estimate of alpha is accepted / (accepted + rejections).
    rejections = sum(1 for _, a, _ in trace if a < k)
    alpha = accepted_total / (accepted_total + rejections) if accepted_total + rejections else None
    avg_draft = ratio(draft_time, rounds * k)
    avg_target = ratio(target_time, rounds)
    c = ratio(avg_draft, avg_target) if avg_target else None

    return new_tokens, {
        "tokens": n,
        "gen_time": gen_time,
        "tok_per_sec": ratio(n, gen_time),
        "prompt_time": prompt_time,
        "k": k,
        "rounds": rounds,
        "proposed": rounds * k,
        "accepted": accepted_total,
        "tokens_per_round": ratio(n, rounds),
        "alpha": alpha,
        "c": c,
        "trace": trace,
    }


# ------------------------------------------------------------
# The model pair, ready to use
# ------------------------------------------------------------

class Engine:
    def __init__(self, draft_name, target_name, device, dtype):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(target_name)
        other = AutoTokenizer.from_pretrained(draft_name)
        if self.tokenizer.get_vocab() != other.get_vocab():
            sys.exit("The draft and target models use different tokenizers. "
                     "Pick two models from the same family.")
        # Output layers can be padded wider than the real vocabulary, so every
        # set of logits is cut to the tokenizer's size before it is compared.
        self.vocab = len(self.tokenizer)

        print("Loading draft model: ", draft_name, flush=True)
        self.draft = load_model(draft_name, device, dtype)
        print("Loading target model:", target_name, flush=True)
        self.target = load_model(target_name, device, dtype)
        self.eos_ids = get_eos_ids(self.tokenizer, self.target)

    def encode(self, prompt):
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"].to(self.device)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def baseline(self, prompt, max_new):
        ids = self.encode(prompt)
        return run_baseline(self.target, ids, self.eos_ids, self.vocab, max_new, self.device)

    def speculative(self, prompt, k, max_new):
        ids = self.encode(prompt)
        return run_speculative(self.draft, self.target, ids, k, self.eos_ids,
                               self.vocab, max_new, self.device)

    def warm_up(self):
        self.baseline("Say hello.", 8)
        self.speculative("Say hello.", 3, 8)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

def token_text(engine, token):
    return repr(engine.tokenizer.decode([token]))


def show_trace(engine, trace, k):
    print("\nRound by round, "
          + paint("green = accepted draft token", "32") + ", "
          + paint("red = rejected", "31;9") + ", "
          + paint("dim = never checked", "2") + ", "
          + paint("cyan = the target's own token", "36;1") + ":")
    marks = not USE_COLOR  # without color, mark tokens with + (accepted), x (rejected), ? (unchecked)
    for i, (proposed, accepted, extra) in enumerate(trace[:TRACE_ROUNDS_SHOWN], start=1):
        parts = []
        for j, token in enumerate(proposed):
            label = token_text(engine, token)
            if j < accepted:
                parts.append(paint(label + ("+" if marks else ""), "32"))
            elif j == accepted:
                parts.append(paint(label + ("x" if marks else ""), "31;9"))
            else:
                parts.append(paint(label + ("?" if marks else ""), "2"))
        extra_label = token_text(engine, extra) + (" (bonus)" if accepted == k else "")
        print(f"  {i:>2}  " + " ".join(parts)
              + "  ->  " + paint(extra_label, "36;1")
              + f"   +{accepted + 1} token{'s' if accepted else ''}")
    if len(trace) > TRACE_ROUNDS_SHOWN:
        print(f"  ... {len(trace) - TRACE_ROUNDS_SHOWN} more rounds")


def fmt_opt(value, fmt):
    return format(value, fmt) if value is not None else "n/a"


def show_comparison(base, spec):
    print("\n" + f"{'':<24}{'baseline':>12}{'speculative':>14}")
    print(f"{'tokens generated':<24}{base['tokens']:>12}{spec['tokens']:>14}")
    print(f"{'generation time (s)':<24}{base['gen_time']:>12.2f}{spec['gen_time']:>14.2f}")
    print(f"{'tokens per second':<24}{base['tok_per_sec']:>12.1f}{spec['tok_per_sec']:>14.1f}")
    speedup = ratio(spec["tok_per_sec"], base["tok_per_sec"])
    print(paint(f"\nSpeedup: {speedup:.2f}x", "1"))
    show_spec_details(spec, speedup)


def show_spec_details(spec, measured_speedup=None):
    pct = ratio(spec["accepted"], spec["proposed"])
    print(f"Draft tokens accepted: {spec['accepted']} of {spec['proposed']} proposed ({pct:.0%}), "
          f"{spec['tokens_per_round']:.2f} tokens per target pass (K = {spec['k']}).")
    theory = predicted_speedup(spec["alpha"], spec["k"], spec["c"])
    print(f"Estimated alpha: {fmt_opt(spec['alpha'], '.2f')}, "
          f"cost ratio c: {fmt_opt(spec['c'], '.2f')} "
          f"(one draft pass costs about {fmt_opt(spec['c'], '.0%')} of a target pass).")
    if theory is not None and measured_speedup is not None:
        print(f"The formula predicts about {theory:.2f}x; you measured {measured_speedup:.2f}x.")
    elif theory is not None:
        print(f"The formula predicts about {theory:.2f}x over the baseline.")


def run_sweep(engine, prompt, max_new):
    print(f"\nSweep over K = {SWEEP_KS} on: {prompt[:70]!r}")
    print("Running baseline...", flush=True)
    _, base = engine.baseline(prompt, max_new)
    rows = []
    for k in SWEEP_KS:
        print(f"Running K = {k}...", flush=True)
        _, spec = engine.speculative(prompt, k, max_new)
        speedup = ratio(spec["tok_per_sec"], base["tok_per_sec"])
        rows.append((k, spec, speedup))

    best = max(s for _, _, s in rows)
    print(f"\nBaseline: {base['tok_per_sec']:.1f} tokens per second\n")
    print(f"{'K':>3}{'tok/s':>9}{'speedup':>9}{'accepted':>10}{'tok/round':>11}"
          f"{'alpha':>7}{'c':>6}{'theory':>8}  chart")
    for k, spec, speedup in rows:
        theory = predicted_speedup(spec["alpha"], k, spec["c"])
        bar = "#" * max(1, round(20 * speedup / best)) if best > 0 else ""
        marker = " <- best" if speedup == best else ""
        print(f"{k:>3}{spec['tok_per_sec']:>9.1f}{speedup:>8.2f}x"
              f"{ratio(spec['accepted'], spec['proposed']):>10.0%}"
              f"{spec['tokens_per_round']:>11.2f}"
              f"{fmt_opt(spec['alpha'], '.2f'):>7}{fmt_opt(spec['c'], '.2f'):>6}"
              f"{fmt_opt(theory, '.2f') + 'x':>8}  {bar}{marker}")
    print("\nA longer draft gives more tokens per pass but costs more draft passes, "
          "so the speedup rises and then falls.")


# ------------------------------------------------------------
# One prompt
# ------------------------------------------------------------

def answer_prompt(engine, prompt, mode, k, max_new, show):
    if mode in ("compare", "baseline"):
        print("Running baseline...", flush=True)
        base_tokens, base = engine.baseline(prompt, max_new)
    if mode in ("compare", "speculative"):
        print("Running speculative...", flush=True)
        spec_tokens, spec = engine.speculative(prompt, k, max_new)

    if mode == "baseline":
        print("\n" + engine.decode(base_tokens))
        print(f"\n{base['tokens']} tokens in {base['gen_time']:.2f} s "
              f"({base['tok_per_sec']:.1f} tokens per second)")
        return

    if mode == "speculative":
        print("\n" + engine.decode(spec_tokens))
        print(f"\n{spec['tokens']} tokens in {spec['gen_time']:.2f} s "
              f"({spec['tok_per_sec']:.1f} tokens per second)")
        show_spec_details(spec)
    else:
        print("\n" + engine.decode(base_tokens))
        if spec_tokens == base_tokens:
            print(paint("\nSpeculative decoding produced exactly the same text.", "32"))
        else:
            print(paint("\nThe speculative text differed slightly from the baseline:", "33"))
            print(engine.decode(spec_tokens))
            print("(This can happen when two tokens are nearly tied, because half-precision "
                  "math rounds a little differently when several tokens are checked at once.)")
        show_comparison(base, spec)

    if show and mode in ("compare", "speculative"):
        show_trace(engine, spec["trace"], k)


# ------------------------------------------------------------
# Command line and interactive loop
# ------------------------------------------------------------

HELP = """
Type a prompt and press Enter. Commands:
  :mode compare | baseline | speculative   what to run (default: compare)
  :k <n>            draft length, tokens guessed per round
  :max <n>          maximum new tokens per answer
  :show on | off    show every round of guesses (green = accepted, red = rejected)
  :sweep [prompt]   try K = 1, 2, 3, 5, 8 and chart the speedups
  :examples         list sample prompts, one per kind of task
  :example <n>      run sample prompt number n
  :help             show this list
  :quit             exit
"""


def parse_args():
    pair_help = "; ".join(f"{name}: {p['note']}" for name, p in PAIRS.items())
    parser = argparse.ArgumentParser(
        description="Try speculative decoding on your own computer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model pairs: " + pair_help,
    )
    parser.add_argument("--pair", choices=PAIRS, default=DEFAULT_PAIR,
                        help=f"model pair to use (default: {DEFAULT_PAIR})")
    parser.add_argument("--draft", help="Hugging Face name of a custom draft model")
    parser.add_argument("--target", help="Hugging Face name of a custom target model")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"draft length (default: {DEFAULT_K})")
    parser.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW,
                        help=f"maximum new tokens per answer (default: {DEFAULT_MAX_NEW})")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="where to run the models (default: auto)")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="number format (default: float16 on a GPU, float32 on a CPU)")
    parser.add_argument("--prompt", help="answer this prompt once and exit")
    parser.add_argument("--show", action="store_true",
                        help="show every round of guesses")
    args = parser.parse_args()
    if bool(args.draft) != bool(args.target):
        parser.error("--draft and --target must be given together")
    if args.k < 1 or args.max_new < 1:
        parser.error("--k and --max-new must be at least 1")
    return args


def main():
    args = parse_args()
    if args.draft:
        draft_name, target_name = args.draft, args.target
    else:
        draft_name, target_name = PAIRS[args.pair]["draft"], PAIRS[args.pair]["target"]

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(f"Device: {device.type}, number format: {str(dtype).replace('torch.', '')}")
    if device.type == "cpu":
        print("No GPU found, so this will be slow. Try --pair tiny and a small --max-new.")

    try:
        engine = Engine(draft_name, target_name, device, dtype)
    except Exception as error:  # download, login, and memory problems end up here
        message = str(error)
        print(f"\nCould not load the models: {message[:300]}")
        if "gated" in message.lower() or "401" in message or "403" in message:
            print("These models need a Hugging Face login. Run `hf auth login`, "
                  "accept the model license on its web page, or use --pair small.")
        elif "memory" in message.lower():
            print("Not enough memory. Try --pair tiny.")
        sys.exit(1)

    print("Warming up...", flush=True)
    engine.warm_up()

    mode, k, max_new = "compare", args.k, args.max_new
    show = args.show
    last_prompt = EXAMPLES[0][1]

    if args.prompt:
        answer_prompt(engine, args.prompt, mode, k, max_new, show)
        return

    print(f"\nReady: {draft_name} drafting for {target_name}.")
    print(HELP)

    while True:
        try:
            line = input(f"\n[{mode}, K={k}, max={max_new}] prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        try:
            if line.startswith(":"):
                parts = line[1:].split(maxsplit=1)
                command = parts[0].lower() if parts else ""
                argument = parts[1].strip() if len(parts) > 1 else ""

                if command in ("quit", "q", "exit"):
                    break
                elif command == "help":
                    print(HELP)
                elif command == "mode" and argument in MODES:
                    mode = argument
                elif command == "k" and argument.isdigit() and int(argument) >= 1:
                    k = int(argument)
                elif command == "max" and argument.isdigit() and int(argument) >= 1:
                    max_new = int(argument)
                elif command == "show" and argument in ("on", "off"):
                    show = argument == "on"
                elif command == "sweep":
                    last_prompt = argument or last_prompt
                    run_sweep(engine, last_prompt, max_new)
                elif command == "examples":
                    print()
                    for number, (name, text) in enumerate(EXAMPLES, start=1):
                        print(f"  {number}. {name:<12}{text[:60].replace(chr(10), ' ')}"
                              f"{'...' if len(text) > 60 else ''}")
                elif command == "example" and argument.isdigit() and 1 <= int(argument) <= len(EXAMPLES):
                    last_prompt = EXAMPLES[int(argument) - 1][1]
                    print(f"Prompt: {last_prompt}")
                    answer_prompt(engine, last_prompt, mode, k, max_new, show)
                else:
                    print("Unrecognized command.")
                    print(HELP)
                continue

            last_prompt = line
            answer_prompt(engine, line, mode, k, max_new, show)
        except KeyboardInterrupt:
            print("\nInterrupted.")

    print("Goodbye.")


if __name__ == "__main__":
    main()
