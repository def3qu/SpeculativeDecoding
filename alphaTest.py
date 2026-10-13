"""
alphaTest.py

Measures the speculative decoding acceptance rate (alpha) for a draft/target
model pair, following Leviathan, Kalman & Matias (2023).

For every prompt and temperature:
  1. The target model generates a continuation (with a KV cache).
  2. Both models score that same continuation in one teacher-forced pass.
  3. At each position i, with p = target distribution and q = draft distribution,
         beta_i = sum over tokens x of min(p(x), q(x))
     which is the probability the draft's token is accepted at that position.
     Alpha is the average of beta_i over all positions.

Because the contexts come from the target model, they follow the same
distribution speculative decoding produces, so this is the alpha that appears
in the speedup formula. At temperature 0 the distributions are one-hot, and
alpha reduces to the fraction of positions where the two models' argmax
tokens agree.

Outputs:
  alpha_<pair>_<set>_<timestamp>.csv          one row per sample
  alpha_summary_<pair>_<set>_<timestamp>.csv  alpha by task and temperature,
                                              with predicted tokens per target pass

Choose the pair with MODEL_PAIR and the prompts with PROMPT_SET below:
  "builtin"    the 20 prompts from speculativeTest.py, for pairing alpha
               with measured speedups
  "specbench"  the 480 Spec-Bench prompts (spec_bench_questions.jsonl), the
               standard benchmark for speculative decoding
"""

import csv
import json
import time
import zlib
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Draft and target must share a tokenizer, since token ids are compared directly.
MODEL_PAIRS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    "llama": ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
}

MODEL_PAIR = "llama"

# Which prompts to use: "builtin" or "specbench".
PROMPT_SET = "specbench"

# Path to the Spec-Bench question file (used only when PROMPT_SET = "specbench").
SPEC_BENCH_FILE = "spec_bench_questions.jsonl"

# Use only the first N prompts of each task, for quick tests. None uses all.
MAX_PROMPTS_PER_TASK = None

MAX_NEW_TOKENS = 100

# Temperature 0 is greedy decoding (what speculativeTest.py uses).
# Higher temperatures show how alpha changes under sampling.
# For Spec-Bench, [0.0] alone is 480 samples; all three is about 3,400.
TEMPERATURES = [0.0, 0.7, 1.0]
#TEMPERATURES = [0.0]
# Samples per prompt at each temperature above 0. Temperature 0 is
# deterministic, so it always uses one sample.
SAMPLES_PER_PROMPT = 3

# Draft lengths for the predicted tokens-per-pass column in the summary.
K_VALUES = [1, 2, 3, 5, 8]

SEED = 1234


PROMPTS = {
    "code": [
        "Write a Python function that checks whether a string is a palindrome.",
        "Write a Python function that returns the nth Fibonacci number using iteration.",
        "Write a SQL query that finds the five highest-paid employees in an employees table.",
        "Write a JavaScript function that removes duplicate values from an array.",
        "Write a Python class for a stack with push, pop, and peek methods.",
    ],
    "factual": [
        "Explain how a binary search tree works.",
        "Explain the difference between TCP and UDP.",
        "Explain what a hash table is and why lookups are fast.",
        "Explain how photosynthesis works.",
        "Explain what causes the seasons on Earth.",
    ],
    "summarization": [
        "Summarize the following text in two sentences:\n\n"
        "The town council met on Tuesday to discuss the proposed expansion of the public "
        "library. Supporters argued that the current building is too small for the growing "
        "number of children's programs and that the computer lab is always full. Opponents "
        "raised concerns about the cost, which is estimated at four million dollars, and "
        "about losing parking spaces downtown. The council agreed to hold a public hearing "
        "next month before voting.",

        "Summarize the following text in two sentences:\n\n"
        "Version control systems track changes to files over time. Developers can see who "
        "changed what and when, return to earlier versions, and work on separate branches "
        "without interfering with each other. When a feature is finished, its branch is "
        "merged back into the main line of development. Git is the most widely used version "
        "control system today, and services such as GitHub host Git repositories online.",

        "Summarize the following text in two sentences:\n\n"
        "Honeybees communicate the location of food through a behavior called the waggle "
        "dance. A forager returning to the hive moves in a figure-eight pattern, and the "
        "angle of the straight middle run relative to vertical indicates the direction of "
        "the food relative to the sun. The duration of the waggle indicates the distance. "
        "Other bees follow the dancer and then fly out to find the source themselves.",

        "Summarize the following text in two sentences:\n\n"
        "The company reported that quarterly revenue rose eight percent compared with the "
        "same period last year, driven mainly by strong sales of its subscription software. "
        "Hardware sales declined slightly because of supply shortages. Operating costs grew "
        "faster than expected after the company hired several hundred new engineers. "
        "Executives said they expect growth to continue but warned that margins may narrow.",

        "Summarize the following text in two sentences:\n\n"
        "Unit tests check small pieces of code, such as a single function, in isolation. "
        "They run quickly and help developers catch mistakes as soon as they are made. "
        "Integration tests check that several components work correctly together, such as "
        "an application and its database. A healthy test suite usually has many unit tests "
        "and a smaller number of slower integration tests.",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Write a poem about the first snowfall in the mountains.",
        "Describe an imaginary city where it is always night.",
        "Write the opening paragraph of a mystery novel set on a train.",
        "Invent a new holiday and describe how people celebrate it.",
    ],
}


# Spec-Bench's multi-turn task is MT-Bench, stored under eight subcategories
# of 10 prompts each. They are grouped into one task of 80 for the summary,
# as in the Spec-Bench papers; the subcategory is kept in the category column.
MT_BENCH_CATEGORIES = {
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
}


SAMPLE_FIELDS = [
    "timestamp", "draft_model", "target_model", "prompt_set",
    "task", "category", "prompt_id", "temperature", "sample",
    "prompt_tokens", "generated_tokens",
    "alpha", "greedy_agreement", "beta_sum", "greedy_matches",
    "seconds",
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


def load_model(name, device):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(device)
    model.eval()
    return model


def encode(tokenizer, prompt, device):
    """Use the chat template for instruct models, raw text otherwise."""
    if tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids.to(device)


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


def check_finite(logits, model_name):
    """fp16 can overflow on some models. Fail loudly rather than record bad alphas."""
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            f"{model_name} produced NaN or inf logits in float16. "
            "Try loading the models with dtype=torch.bfloat16 in load_model()."
        )


def to_probs(logits, temperature):
    """Temperature 0 gives a one-hot distribution on the argmax token."""
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def load_prompts():
    """Returns a list of dicts with task, category, prompt_id, and prompt."""
    prompts = []

    if PROMPT_SET == "builtin":
        # prompt_id matches speculativeTest.py, so the CSVs can be joined.
        for category, texts in PROMPTS.items():
            for p_index, text in enumerate(texts, start=1):
                prompts.append({
                    "task": category,
                    "category": category,
                    "prompt_id": f"{category}-{p_index}",
                    "prompt": text,
                })

    elif PROMPT_SET == "specbench":
        with open(SPEC_BENCH_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                category = record["category"]
                prompts.append({
                    "task": "mt_bench" if category in MT_BENCH_CATEGORIES else category,
                    "category": category,
                    "prompt_id": f"sb-{record['question_id']}",
                    # MT-Bench questions have a second turn; only the first is used.
                    "prompt": record["turns"][0],
                })

    else:
        raise SystemExit(f'Unknown PROMPT_SET {PROMPT_SET!r}. Use "builtin" or "specbench".')

    if MAX_PROMPTS_PER_TASK is not None:
        kept, counts = [], defaultdict(int)
        for p in prompts:
            if counts[p["task"]] < MAX_PROMPTS_PER_TASK:
                kept.append(p)
                counts[p["task"]] += 1
        prompts = kept

    return prompts


def predicted_tokens_per_pass(alpha, k):
    """Expected tokens per target forward pass: (1 - alpha^(K+1)) / (1 - alpha)."""
    if alpha >= 1.0:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


# ------------------------------------------------------------
# Step 1: target model generates the continuation
# ------------------------------------------------------------

@torch.no_grad()
def target_continuation(model, model_name, input_ids, temperature, vocab, eos_ids, gen):
    """Plain autoregressive generation with a KV cache. Written by hand rather
    than with model.generate() so that no default top-k, top-p, or repetition
    settings from the model's generation_config change the distribution."""
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :vocab].float().cpu()
    check_finite(logits, model_name)

    tokens = []
    while True:
        if temperature == 0:
            token = int(logits.argmax())
        else:
            probs = to_probs(logits, temperature)
            token = int(torch.multinomial(probs, 1, generator=gen).item())
        tokens.append(token)

        if token in eos_ids or len(tokens) >= MAX_NEW_TOKENS:
            break

        out = model(
            input_ids=torch.tensor([[token]], device=input_ids.device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[0, -1, :vocab].float().cpu()

    return tokens


# ------------------------------------------------------------
# Step 2: both models score the same continuation
# ------------------------------------------------------------

@torch.no_grad()
def teacher_forced_logits(model, model_name, full_ids, prompt_len, vocab):
    """Logits that predict each generated token. Row i predicts token
    prompt_len + i, so it comes from position prompt_len + i - 1."""
    logits = model(input_ids=full_ids).logits[0, prompt_len - 1:-1, :vocab]
    logits = logits.float().cpu()
    check_finite(logits, model_name)
    return logits


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

def main():
    device = pick_device()
    draft_name, target_name = MODEL_PAIRS[MODEL_PAIR]

    print("Using device:", device)
    print("Draft model: ", draft_name)
    print("Target model:", target_name)

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise SystemExit("Draft and target tokenizers differ. Pick models that share a vocabulary.")

    # The models' output layers can be padded wider than the real vocabulary
    # (Qwen2.5 0.5B and 7B are padded to different widths), so every logits
    # tensor is cut to the tokenizer's size before the models are compared.
    vocab = len(tokenizer)

    print("Loading draft model...")
    draft_model = load_model(draft_name, device)
    print("Loading target model...")
    target_model = load_model(target_name, device)

    eos_ids = get_eos_ids(tokenizer, target_model)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    samples_path = f"alpha_{MODEL_PAIR}_{PROMPT_SET}_{stamp}.csv"
    summary_path = f"alpha_summary_{MODEL_PAIR}_{PROMPT_SET}_{stamp}.csv"

    prompts = load_prompts()
    print(f"Prompt set: {PROMPT_SET} ({len(prompts)} prompts)")

    jobs = []
    for p in prompts:
        for temperature in TEMPERATURES:
            n_samples = 1 if temperature == 0 else SAMPLES_PER_PROMPT
            for sample in range(1, n_samples + 1):
                jobs.append((p, temperature, sample))

    # Running totals for the summary, keyed by (task, temperature).
    totals = defaultdict(lambda: {"samples": 0, "tokens": 0, "beta_sum": 0.0, "greedy": 0.0})

    start = time.perf_counter()

    with open(samples_path, "w", newline="") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()

        for job_index, (p_info, temperature, sample) in enumerate(jobs, start=1):
            t0 = time.perf_counter()
            task = p_info["task"]
            prompt_id = p_info["prompt_id"]
            prompt = p_info["prompt"]

            # A fixed seed per sample makes the run reproducible.
            key = f"{prompt_id}|{temperature}|{sample}"
            gen = torch.Generator().manual_seed(SEED + zlib.crc32(key.encode()))

            input_ids = encode(tokenizer, prompt, device)
            prompt_len = input_ids.shape[1]

            # Step 1: target generates.
            continuation = target_continuation(
                target_model, target_name, input_ids, temperature, vocab, eos_ids, gen
            )
            full_ids = torch.cat(
                [input_ids, torch.tensor([continuation], dtype=torch.long, device=device)],
                dim=1,
            )

            # Step 2: both models score the same tokens.
            target_logits = teacher_forced_logits(target_model, target_name, full_ids, prompt_len, vocab)
            draft_logits = teacher_forced_logits(draft_model, draft_name, full_ids, prompt_len, vocab)

            # Step 3: beta_i = sum_x min(p(x), q(x)) at every position.
            p = to_probs(target_logits, temperature)
            q = to_probs(draft_logits, temperature)
            betas = torch.minimum(p, q).sum(dim=-1)
            greedy = (target_logits.argmax(-1) == draft_logits.argmax(-1)).float()

            n = len(continuation)
            beta_sum = float(betas.sum())
            greedy_sum = float(greedy.sum())
            seconds = time.perf_counter() - t0

            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "draft_model": draft_name,
                "target_model": target_name,
                "prompt_set": PROMPT_SET,
                "task": task,
                "category": p_info["category"],
                "prompt_id": prompt_id,
                "temperature": temperature,
                "sample": sample,
                "prompt_tokens": prompt_len,
                "generated_tokens": n,
                "alpha": beta_sum / n,
                "greedy_agreement": greedy_sum / n,
                "beta_sum": beta_sum,
                "greedy_matches": greedy_sum,
                "seconds": round(seconds, 2),
            })
            samples_file.flush()

            for group in (task, "ALL"):
                t = totals[(group, temperature)]
                t["samples"] += 1
                t["tokens"] += n
                t["beta_sum"] += beta_sum
                t["greedy"] += greedy_sum

            elapsed = time.perf_counter() - start
            eta = elapsed / job_index * (len(jobs) - job_index)
            print(
                f"[{job_index}/{len(jobs)}] {prompt_id} T={temperature} s{sample}: "
                f"{n} tokens, alpha={beta_sum / n:.3f} "
                f"({seconds:.1f}s, ETA {eta / 60:.1f} min)"
            )

    # --------------------------------------------------------
    # Summary: token-weighted alpha by task and temperature
    # --------------------------------------------------------

    summary_fields = ["task", "temperature", "samples", "tokens", "alpha", "greedy_agreement"]
    summary_fields += [f"pred_tokens_per_pass_K{k}" for k in K_VALUES]

    def sort_key(item):
        (group, temperature), _ = item
        return (temperature, group == "ALL", group)

    rows = []
    for (group, temperature), t in sorted(totals.items(), key=sort_key):
        alpha = t["beta_sum"] / t["tokens"]
        row = {
            "task": group,
            "temperature": temperature,
            "samples": t["samples"],
            "tokens": t["tokens"],
            "alpha": round(alpha, 4),
            "greedy_agreement": round(t["greedy"] / t["tokens"], 4),
        }
        for k in K_VALUES:
            row[f"pred_tokens_per_pass_K{k}"] = round(predicted_tokens_per_pass(alpha, k), 3)
        rows.append(row)

    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'task':<15}{'temp':>6}{'n':>5}{'tokens':>8}{'alpha':>8}{'greedy':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['task']:<15}{r['temperature']:>6.1f}{r['samples']:>5}{r['tokens']:>8}"
            f"{r['alpha']:>8.3f}{r['greedy_agreement']:>8.3f}"
        )

    print(f"\nFinished in {(time.perf_counter() - start) / 60:.1f} minutes")
    print("Samples written to", samples_path)
    print("Summary written to", summary_path)


if __name__ == "__main__":
    main()
