#!/usr/bin/env python3
"""
measure_alpha.py

Estimate the speculative decoding acceptance rate (alpha) for a draft/target
model pair, following Leviathan, Kalman & Matias (2023), "Fast Inference from
Transformers via Speculative Decoding."

Two estimates are produced for every sample:

1. Analytic alpha (always computed).
   The target model generates a continuation. Both models then score that
   continuation in a single teacher-forced pass. At each position
       beta = sum_x min(p(x), q(x))      (p = target, q = draft)
   which equals 1 - TV(p, q). Alpha is the mean of beta over positions.
   Because the contexts are produced by the target model, they follow the same
   distribution speculative decoding produces, so this is the alpha that
   appears in the speedup formula. At temperature 0 it reduces to the
   fraction of positions where the two models' argmax tokens agree.

2. Empirical alpha (optional, on by default).
   A real speculative sampling loop is run with draft length gamma.
   Empirical alpha = accepted draft tokens / tested draft tokens.
   We also record the mean number of tokens produced per target forward pass
   and compare it with the theoretical prediction (1 - a^(g+1)) / (1 - a).
   This loop recomputes the target forward pass without a KV cache, so it is
   accurate for counting acceptances but should NOT be used for timing.

Results are appended to a JSONL file after every sample. If the script is
stopped (Ctrl-C, reboot, lost SSH session), rerun the same command and
finished samples are skipped.

Typical use:
    # quick smoke test (a few minutes)
    python measure_alpha.py --pair smol --max-prompts 2 --samples-per-prompt 1 --max-new-tokens 32

    # full unattended run
    nohup python measure_alpha.py --pair smol > smol_run.log 2>&1 &

    # Llama pairs (gated models: run `hf auth login` first)
    nohup python measure_alpha.py --pair llama-1b-3b > llama_small_run.log 2>&1 &
    # 70B target needs a large GPU, or --target-4bit (about 40 GB of VRAM)
    nohup python measure_alpha.py --pair llama-1b --target-4bit > llama_run.log 2>&1 &

    # re-print the summary from existing results without loading models
    python measure_alpha.py --pair smol --summarize-only
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
import zlib
from collections import defaultdict

# ---------------------------------------------------------------------------
# Model pairs. Edit or add entries here. Draft and target must share a tokenizer.
# ---------------------------------------------------------------------------
PRESETS = {
    # SmolLM v1 base models (the pair used in speculative.py)
    "smol": ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-1.7B"),
    # SmolLM2 instruct models (all share one tokenizer)
    "smol2": ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    "smol2-360": ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    # Llama 3 family (Llama 3.2 and 3.3 share the 128k Llama 3 tokenizer)
    "llama-1b": ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
#    "llama-3b": ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"),
    # Small Llama pair, useful for testing the Llama path on modest hardware
    "llama-1b-3b": ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
}

# ---------------------------------------------------------------------------
# Built-in prompt set, grouped by task type. Replace with --prompts-file
# (JSONL lines of {"id": ..., "category": ..., "prompt": ...}) if you like.
# ---------------------------------------------------------------------------
BUILTIN_PROMPTS = {
    "code": [
        "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Include a docstring.",
        "Write a Python class implementing a stack with push, pop, peek, and is_empty methods.",
        "Write a SQL query that returns the top five customers by total order amount from tables customers(id, name) and orders(id, customer_id, amount).",
        "Write a JavaScript function that debounces another function by a given number of milliseconds.",
        "Write a Python function that performs binary search on a sorted list and returns the index of the target or -1.",
    ],
    "math": [
        "Solve step by step: A train travels 240 miles in 4 hours. At the same speed, how long will it take to travel 420 miles?",
        "Explain step by step how to find the derivative of f(x) = x^3 * sin(x).",
        "A rectangle has a perimeter of 36 cm and its length is twice its width. Find its area, showing your work.",
        "Prove that the sum of the first n odd numbers is n squared.",
        "What is the probability of rolling at least one six in four rolls of a fair die? Show the calculation.",
    ],
    "factual": [
        "Explain how a hash table works and why lookups are usually constant time.",
        "Describe the causes of the seasons on Earth.",
        "Explain the difference between TCP and UDP.",
        "What is photosynthesis? Explain the main stages.",
        "Explain what a compiler does, from source code to machine code.",
    ],
    "summarize": [
        "Summarize in three sentences: The Appalachian Trail is a hiking trail in the eastern United States, running about 2,200 miles between Springer Mountain in Georgia and Mount Katahdin in Maine. It passes through fourteen states. The trail was conceived by Benton MacKaye in 1921 and completed in 1937. Each year several thousand hikers attempt to walk its full length in a single season, though only about one in four succeed. The trail is maintained largely by volunteers organized through local clubs.",
        "Summarize the key points: Version control systems record changes to files over time so that specific versions can be recalled later. Centralized systems keep the full history on a single server, while distributed systems such as Git give every developer a complete copy of the repository. Distributed systems make branching and merging cheap, allow offline work, and remove the server as a single point of failure.",
        "Summarize for a general audience: Large language models generate text one token at a time. For each token, the model reads all of its weights from memory, so on consumer hardware generation speed is limited mostly by memory bandwidth rather than arithmetic. Techniques that produce more than one token per pass over the weights can therefore speed up generation substantially.",
        "Write a one-paragraph summary: Honeybees communicate the location of food through the waggle dance. The angle of the dance relative to vertical indicates the direction of the food relative to the sun, and the duration of the waggle portion indicates distance. Other bees follow the dancer and then fly to the food source. Karl von Frisch received a Nobel Prize in 1973 in part for decoding this behavior.",
        "Summarize in bullet points: Agile software development emphasizes short iterations, frequent delivery of working software, close collaboration with customers, and responsiveness to change. Teams typically plan work in sprints of one to four weeks, hold brief daily stand-up meetings, and reflect on their process in retrospectives at the end of each sprint.",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
        "Write a poem about autumn in the mountains.",
        "Invent a new board game and describe its rules.",
        "Write a dialogue between a robot and a cat who meet for the first time.",
        "Describe an imaginary city built entirely underwater.",
    ],
    "chat": [
        "Hi! I'm thinking about learning to play guitar. Any advice for a complete beginner?",
        "Can you recommend a good routine for staying productive while working from home?",
        "I have a job interview tomorrow for a software developer position. How should I prepare tonight?",
        "What are some fun things to do on a rainy weekend?",
        "I'm trying to eat healthier but I'm busy. What are some easy meal ideas?",
    ],
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Estimate speculative decoding acceptance rate (alpha).")
    ap.add_argument("--pair", default="smol", choices=sorted(PRESETS.keys()),
                    help="Preset draft/target pair (default: smol).")
    ap.add_argument("--draft", help="Override draft model (Hugging Face id or local path).")
    ap.add_argument("--target", help="Override target model (Hugging Face id or local path).")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7, 1.0],
                    help="Sampling temperatures to test (0 = greedy). Default: 0 0.7 1.0")
    ap.add_argument("--samples-per-prompt", type=int, default=3,
                    help="Samples per prompt at each temperature > 0 (temperature 0 always uses 1). Default: 3")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="Tokens generated per sample. Default: 256")
    ap.add_argument("--gamma", type=int, default=5, help="Draft length (K) for the empirical loop. Default: 5")
    ap.add_argument("--no-empirical", action="store_true",
                    help="Skip the empirical speculative loop (much faster for large targets).")
    ap.add_argument("--prompts-file", help="JSONL file with {id, category, prompt} per line.")
    ap.add_argument("--categories", nargs="+", help="Only run these categories.")
    ap.add_argument("--max-prompts", type=int, help="Use only the first N prompts per category (for testing).")
    ap.add_argument("--out-dir", default="alpha_results", help="Output directory. Default: alpha_results")
    ap.add_argument("--run-name", help="Subdirectory name for this run (default: derived from the pair).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--target-4bit", action="store_true", help="Load target in 4-bit (bitsandbytes, CUDA only).")
    ap.add_argument("--target-8bit", action="store_true", help="Load target in 8-bit (bitsandbytes, CUDA only).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--summarize-only", action="store_true", help="Print summary of existing results and exit.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.split("/")[-1])


def load_prompts(args):
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompts.append({
                    "id": str(rec.get("id", f"p{i}")),
                    "category": rec.get("category", "uncategorized"),
                    "prompt": rec["prompt"],
                })
    else:
        for cat, plist in BUILTIN_PROMPTS.items():
            for i, p in enumerate(plist):
                prompts.append({"id": f"{cat}-{i}", "category": cat, "prompt": p})
    if args.categories:
        prompts = [p for p in prompts if p["category"] in args.categories]
    if args.max_prompts:
        kept, counts = [], defaultdict(int)
        for p in prompts:
            if counts[p["category"]] < args.max_prompts:
                kept.append(p)
                counts[p["category"]] += 1
        prompts = kept
    return prompts


def setup_logging(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(run_dir, "run.log"))):
        h.setFormatter(fmt)
        root.addHandler(h)


def load_done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass  # ignore a partially written last line
    return done


def predicted_tokens_per_pass(alpha, gamma):
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Summary (no torch needed)
# ---------------------------------------------------------------------------
def summarize(samples_path, run_dir):
    if not os.path.exists(samples_path):
        print(f"No results at {samples_path}")
        return
    groups = defaultdict(lambda: defaultdict(float))

    def add(key, r):
        g = groups[key]
        g["samples"] += 1
        g["tokens"] += r["n_tokens"]
        g["beta_sum"] += r["beta_sum"]
        g["greedy_match_sum"] += r["greedy_match_sum"]
        if "empirical" in r:
            e = r["empirical"]
            g["emp_accepted"] += e["accepted"]
            g["emp_tested"] += e["tested"]
            g["emp_tokens"] += e["tokens"]
            g["emp_rounds"] += e["rounds"]
            g["gamma"] = e["gamma"]

    with open(samples_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            add((r["category"], r["temperature"]), r)
            add(("ALL", r["temperature"]), r)

    rows = []
    for (cat, temp), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0] == "ALL", kv[0][0])):
        if g["tokens"] == 0:
            continue
        alpha = g["beta_sum"] / g["tokens"]
        row = {
            "category": cat,
            "temperature": temp,
            "samples": int(g["samples"]),
            "tokens": int(g["tokens"]),
            "alpha_analytic": round(alpha, 4),
            "greedy_agreement": round(g["greedy_match_sum"] / g["tokens"], 4),
        }
        if g["emp_tested"] > 0:
            gamma = int(g["gamma"])
            row["gamma"] = gamma
            row["alpha_empirical"] = round(g["emp_accepted"] / g["emp_tested"], 4)
            row["tokens_per_pass_measured"] = round(g["emp_tokens"] / g["emp_rounds"], 3)
            row["tokens_per_pass_predicted"] = round(predicted_tokens_per_pass(alpha, gamma), 3)
        rows.append(row)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    fields = ["category", "temperature", "samples", "tokens", "alpha_analytic", "greedy_agreement",
              "gamma", "alpha_empirical", "tokens_per_pass_measured", "tokens_per_pass_predicted"]
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    hdr = f"{'category':<12}{'temp':>6}{'n':>5}{'tokens':>8}{'alpha':>8}{'greedy':>8}{'emp_a':>8}{'tok/pass':>10}{'pred':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['category']:<12}{r['temperature']:>6.2f}{r['samples']:>5}{r['tokens']:>8}"
              f"{r['alpha_analytic']:>8.3f}{r['greedy_agreement']:>8.3f}"
              f"{r.get('alpha_empirical', float('nan')):>8.3f}"
              f"{r.get('tokens_per_pass_measured', float('nan')):>10.3f}"
              f"{r.get('tokens_per_pass_predicted', float('nan')):>8.3f}")
    print(f"\nSummary written to {run_dir}/summary.csv and summary.json\n")


# ---------------------------------------------------------------------------
# Main measurement code
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    draft_name, target_name = PRESETS[args.pair]
    draft_name = args.draft or draft_name
    target_name = args.target or target_name
    run_name = args.run_name or (args.pair if not (args.draft or args.target)
                                 else f"{slug(draft_name)}__{slug(target_name)}")
    run_dir = os.path.join(args.out_dir, run_name)
    samples_path = os.path.join(run_dir, "samples.jsonl")

    if args.summarize_only:
        summarize(samples_path, run_dir)
        return

    setup_logging(run_dir)
    log = logging.getLogger()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- device and dtype ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    if args.dtype == "auto":
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # ---- config check for resume ----
    config = {
        "draft": draft_name, "target": target_name, "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma, "empirical": not args.no_empirical,
        "target_quant": "4bit" if args.target_4bit else "8bit" if args.target_8bit else "none",
        "dtype": str(dtype), "device": device,
    }
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        for k in ("draft", "target", "max_new_tokens", "gamma", "target_quant"):
            if old.get(k) != config[k]:
                log.error(f"Existing run in {run_dir} used {k}={old.get(k)!r}, now {config[k]!r}. "
                          f"Use a different --run-name or delete the directory.")
                sys.exit(1)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"Draft:  {draft_name}")
    log.info(f"Target: {target_name}")
    log.info(f"Device: {device}, dtype: {dtype}, target quantization: {config['target_quant']}")

    # ---- tokenizer check ----
    tok = AutoTokenizer.from_pretrained(target_name)
    tok_d = AutoTokenizer.from_pretrained(draft_name)
    if tok.get_vocab() != tok_d.get_vocab():
        log.error("Draft and target tokenizers differ; standard speculative decoding requires a shared vocabulary.")
        sys.exit(1)
    vocab = len(tok)  # model output dims may be padded past this; we truncate to the real vocab

    # ---- model loading ----
    def load(name, quant_bits=None):
        kw = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if quant_bits:
            from transformers import BitsAndBytesConfig
            if quant_bits == 4:
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
            else:
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kw["device_map"] = "auto"
        elif device == "cuda":
            kw["device_map"] = "auto"
        m = AutoModelForCausalLM.from_pretrained(name, **kw)
        if "device_map" not in kw:
            m.to(device)
        m.eval()
        return m

    t0 = time.time()
    draft = load(draft_name)
    target = load(target_name, 4 if args.target_4bit else 8 if args.target_8bit else None)
    log.info(f"Models loaded in {time.time() - t0:.1f}s")

    def in_dev(model):
        return model.get_input_embeddings().weight.device

    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    gen_eos = getattr(target.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, int):
        eos_ids.add(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        eos_ids.update(gen_eos)
    log.info(f"EOS token ids: {sorted(eos_ids)}")

    # ---- probability utilities (all on CPU, float32) ----
    def to_probs(logits, temperature):
        """logits: (..., V) float tensor. Temperature 0 gives a one-hot argmax distribution."""
        if temperature == 0:
            p = torch.zeros_like(logits)
            p.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
            return p
        return torch.softmax(logits / temperature, dim=-1)

    def sample_from(probs, gen):
        return int(torch.multinomial(probs, 1, generator=gen).item())

    @torch.no_grad()
    def sample_tokens(model, ids, n, temperature, gen, keep_logits=False):
        """Plain autoregressive sampling with a KV cache. Avoids model.generate so that
        no default top-k / top-p / repetition settings from generation_config sneak in."""
        dev = in_dev(model)
        out = model(input_ids=ids.to(dev), use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :vocab].float().cpu()
        tokens, kept = [], []
        for step in range(n):
            if keep_logits:
                kept.append(logits)
            t = sample_from(to_probs(logits, temperature), gen)
            tokens.append(t)
            if t in eos_ids or step == n - 1:
                break
            out = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :vocab].float().cpu()
        return tokens, kept

    @torch.no_grad()
    def teacher_forced(model, full_ids, n_prompt):
        """Logits predicting each continuation token: rows correspond to tokens n_prompt..end."""
        out = model(input_ids=full_ids.to(in_dev(model)))
        return out.logits[0, n_prompt - 1:-1, :vocab].float().cpu()

    @torch.no_grad()
    def speculative_run(prompt_ids, temperature, gen):
        """Standard speculative sampling (Leviathan et al. 2023, Algorithm 1)."""
        ctx = prompt_ids.clone()
        produced = rounds = accepted = tested = 0
        per_round = []
        finished = False
        while produced < args.max_new_tokens and not finished:
            g = min(args.gamma, args.max_new_tokens - produced)
            d_tokens, d_logits = sample_tokens(draft, ctx, g, temperature, gen, keep_logits=True)
            k = len(d_tokens)
            cand = torch.cat([ctx, torch.tensor([d_tokens], dtype=ctx.dtype)], dim=1)
            t_logits = target(input_ids=cand.to(in_dev(target))).logits[0, ctx.shape[1] - 1:, :vocab].float().cpu()
            p = to_probs(t_logits, temperature)          # k + 1 rows
            q = to_probs(torch.stack(d_logits), temperature)  # k rows
            new, n_acc, rejected = [], 0, False
            for i, t in enumerate(d_tokens):
                tested += 1
                ratio = (p[i, t] / q[i, t]).item() if q[i, t] > 0 else 0.0
                if torch.rand(1, generator=gen).item() < min(1.0, ratio):
                    n_acc += 1
                    new.append(t)
                    if t in eos_ids:
                        finished = True
                        break
                else:
                    resid = torch.clamp(p[i] - q[i], min=0.0)
                    s = resid.sum()
                    new.append(sample_from(resid / s if s > 0 else p[i], gen))
                    rejected = True
                    break
            if not rejected and not finished:
                new.append(sample_from(p[k], gen))
            if any(t in eos_ids for t in new):
                finished = True
            new = new[: args.max_new_tokens - produced]
            ctx = torch.cat([ctx, torch.tensor([new], dtype=ctx.dtype)], dim=1)
            produced += len(new)
            accepted += n_acc
            rounds += 1
            per_round.append(n_acc)
        return {"gamma": args.gamma, "accepted": accepted, "tested": tested, "tokens": produced,
                "rounds": rounds, "accepted_per_round": per_round}

    use_chat = bool(getattr(tok, "chat_template", None))
    log.info(f"Prompt format: {'chat template' if use_chat else 'raw text (no chat template)'}")

    # ---- work list ----
    prompts = load_prompts(args)
    jobs = []
    for pr in prompts:
        for temp in args.temperatures:
            n = 1 if temp == 0 else args.samples_per_prompt
            for s in range(n):
                jobs.append((pr, temp, s))
    done = load_done_keys(samples_path)
    todo = [j for j in jobs if f"{j[0]['id']}|{j[1]}|{j[2]}" not in done]
    log.info(f"{len(prompts)} prompts, {len(jobs)} samples total, {len(jobs) - len(todo)} already done, "
             f"{len(todo)} to run.")

    start = time.time()
    with open(samples_path, "a") as out_f:
        for idx, (pr, temp, s) in enumerate(todo, 1):
            key = f"{pr['id']}|{temp}|{s}"
            seed = args.seed + zlib.crc32(key.encode())
            gen = torch.Generator().manual_seed(seed)
            t_sample = time.time()

            if use_chat:
                text = tok.apply_chat_template([{"role": "user", "content": pr["prompt"]}],
                                               tokenize=False, add_generation_prompt=True)
                prompt_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            else:
                # Base models (e.g. SmolLM v1) have no chat template: use the raw prompt.
                prompt_ids = tok(pr["prompt"] + "\n", return_tensors="pt").input_ids

            # 1) target generates the continuation
            cont, _ = sample_tokens(target, prompt_ids, args.max_new_tokens, temp, gen)
            full = torch.cat([prompt_ids, torch.tensor([cont], dtype=prompt_ids.dtype)], dim=1)

            # 2) teacher-forced scoring by both models
            n_prompt = prompt_ids.shape[1]
            tl = teacher_forced(target, full, n_prompt)
            dl = teacher_forced(draft, full, n_prompt)
            p, q = to_probs(tl, temp), to_probs(dl, temp)
            betas = torch.minimum(p, q).sum(-1)
            greedy = (tl.argmax(-1) == dl.argmax(-1)).float()
            ent_t = temp if temp > 0 else 1.0
            pe = torch.softmax(tl / ent_t, dim=-1)
            entropy = -(pe * torch.log(pe.clamp_min(1e-12))).sum(-1)

            rec = {
                "key": key, "prompt_id": pr["id"], "category": pr["category"], "temperature": temp,
                "sample": s, "seed": seed, "n_tokens": len(cont),
                "beta_sum": float(betas.sum()), "alpha_analytic": float(betas.mean()),
                "greedy_match_sum": float(greedy.sum()),
                "betas": [round(float(b), 4) for b in betas],
                "target_entropy": [round(float(e), 4) for e in entropy],
                "continuation": tok.decode(cont, skip_special_tokens=True),
            }

            # 3) optional empirical speculative loop
            if not args.no_empirical:
                rec["empirical"] = speculative_run(prompt_ids, temp, gen)

            rec["seconds"] = round(time.time() - t_sample, 2)
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            elapsed = time.time() - start
            eta = elapsed / idx * (len(todo) - idx)
            msg = (f"[{idx}/{len(todo)}] {key:<22} tokens={len(cont):>4} alpha={rec['alpha_analytic']:.3f}")
            if "empirical" in rec:
                e = rec["empirical"]
                emp = e["accepted"] / e["tested"] if e["tested"] else float("nan")
                msg += f" emp_alpha={emp:.3f} tok/pass={e['tokens'] / max(e['rounds'], 1):.2f}"
            msg += f"  ({rec['seconds']:.1f}s, ETA {eta / 3600:.2f}h)"
            log.info(msg)

    log.info(f"Finished in {(time.time() - start) / 3600:.2f}h")
    summarize(samples_path, run_dir)


if __name__ == "__main__":
    main()
