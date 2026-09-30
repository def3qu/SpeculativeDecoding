# Speculative Decoding on Consumer Hardware

Code for testing whether the analytic model of speculative decoding (Leviathan et al., 2023) holds on consumer-grade hardware, using open-weight models on Apple Silicon.

Speculative decoding has a small **draft** model guess several tokens ahead, then has a large **target** model check all of the guesses in one forward pass. Correct guesses are kept, the first wrong one is replaced by the target's own token, and the process repeats. With greedy decoding, the output is identical to the target model running alone; only the speed changes.

The theory predicts the speedup from three quantities:

| Symbol | Meaning |
| --- | --- |
| α (alpha) | Per-token acceptance rate: how often the draft's guess matches the target |
| c | Cost of one draft forward pass relative to one target forward pass |
| K | Number of tokens the draft guesses per round |

```
expected tokens per round   E = (1 - α^(K+1)) / (1 - α)
predicted speedup           S = E / (c·K + 1)
```

## Programs

| File | Purpose | Output |
| --- | --- | --- |
| `speculative.py` | Runs one experiment: baseline vs. speculative decoding on one prompt at one K | Console report |
| `sweep.py` | Runs the experiment across task types and K values | Timestamped CSV |
| `trace_rounds.py` | Shows what happens token by token in a few rounds | Console tables and `trace.html` |

## Setup

**Requirements:** Python 3.10 or later, PyTorch, and Hugging Face Transformers.

```bash
pip install torch transformers
```

The scripts use Apple Silicon (MPS) when available, then CUDA, then CPU.

**Model access:** the default pair, Llama 3.2 1B and 3B Instruct, is gated. Request access on each model's Hugging Face page, then authenticate once:

```bash
huggingface-cli login
# or
export HF_TOKEN=your_token_here
```

Weights download on first run and are cached in `~/.cache/huggingface/hub`.

**Changing the model pair:** edit `DRAFT_MODEL` and `TARGET_MODEL` at the top of `speculative.py`. The other scripts import these settings, so one change applies everywhere. Requirements for a pair:

- Both models must share a tokenizer, which in practice means the same model family and generation.
- Both must be Instruct (chat) models, because the scripts format prompts with the chat template. For the SmolLM family, use `HuggingFaceTB/SmolLM-135M-Instruct` and `HuggingFaceTB/SmolLM-1.7B-Instruct`, not the base models.

---

## speculative.py

The core experiment and a shared library for the other scripts.

**What it does:** runs greedy decoding with the target model alone (the baseline) and greedy speculative decoding with the draft and target together, on the same prompt, with KV caching in both. Each method runs once as a warmup, then `TRIALS` timed runs are averaged. It checks that both methods produce identical output, then reports measured and predicted performance.

**Usage:**

```bash
python speculative.py
```

**Configuration** (top of file):

| Setting | Default | Meaning |
| --- | --- | --- |
| `DRAFT_MODEL`, `TARGET_MODEL` | Llama 3.2 1B / 3B Instruct | Model pair |
| `PROMPT` | "Explain how a binary search tree works." | Prompt for a single run |
| `MAX_NEW_TOKENS` | 100 | Tokens generated per run |
| `K` | 5 | Draft tokens guessed per round |
| `TRIALS` | 3 | Timed runs per method |
| `DTYPE` | `torch.float16` | Model precision |

**Output:**

- Whether speculative output matches the baseline exactly. Rare mismatches late in the text can come from float16 rounding.
- Baseline and speculative tokens per second, and the measured speedup.
- Predicted speedup from the formula, using the measured α and c.
- α, c, and the **verify ratio**: the cost of verifying K+1 tokens divided by the cost of one normal decoding step. A value near 1 shows verification is nearly free, which is the memory-bandwidth effect speculative decoding relies on.
- Measured vs. predicted tokens per round.
- A histogram of accepted tokens per round, compared with the distribution the theory predicts.

**Main functions** (used by the other scripts):

| Function | Consumes | Produces |
| --- | --- | --- |
| `encode(prompt)` | Prompt text | Token ids in chat format, shape (1, n) |
| `step(model, cache, seq)` | Model, its KV cache, full token sequence | Logits for the tokens the cache has not seen; updates the cache |
| `run_baseline(prompt_ids, max_new)` | Prompt ids, token budget | Generated ids, total time, per-step times |
| `run_speculative(prompt_ids, max_new, k)` | Prompt ids, token budget, K | Generated ids, total time, accepted count per round, timings |
| `summarize(base, spec, k)` | Lists of run results, K | Dictionary of all metrics |
| `warmup()` | Nothing | Runs both methods briefly so GPU kernels compile before timing |

---

## sweep.py

The main data-collection script.

**What it does:** loads the models once, then for each task prompt measures the baseline and runs speculative decoding at every value in `K_VALUES`. Each (task, K) combination becomes one CSV row.

**Usage:**

```bash
python sweep.py
```

Expect roughly 10 to 20 minutes with the defaults on a Mac Studio. Rows are written as they complete, so an interrupted run keeps its partial results.

**Configuration:**

| Setting | Default | Meaning |
| --- | --- | --- |
| `TASKS` | code, explanation, math, summarization, creative, structured | Prompt for each task type |
| `K_VALUES` | 1, 2, 3, 4, 5, 6, 8 | Draft lengths to test |
| `TRIALS` | 3 | Timed runs per configuration |
| `MAX_NEW_TOKENS` | 128 | Tokens generated per run |

To add a task, add an entry to the `TASKS` dictionary. One prompt per task is enough to test the pipeline; use several per task for results you plan to report.

**Output:** `sweep_YYYYMMDD_HHMMSS.csv` with these columns:

| Column | Meaning |
| --- | --- |
| `timestamp`, `device`, `draft_model`, `target_model` | Run metadata, so CSVs from different model pairs can be combined |
| `task`, `k` | Configuration |
| `baseline_tps`, `spec_tps` | Tokens per second for each method |
| `measured_speedup`, `predicted_speedup` | Observed vs. theoretical speedup |
| `alpha`, `c`, `verify_ratio` | Inputs to the theory, as measured |
| `tokens_per_iter_measured`, `tokens_per_iter_predicted` | Observed vs. theoretical tokens per round |
| `iterations` | Total rounds across all trials |
| `outputs_match`, `first_diff` | Whether speculative output matched the baseline |
| `accept_hist` | Accepted-per-round counts, e.g. `0:12;1:5;2:8;...` |

---

## trace_rounds.py

A teaching and presentation tool.

**What it does:** runs a few rounds of speculative decoding on one prompt and records every token: what the draft guessed, what the target wanted at the same position, and which tokens were kept.

**Usage:**

```bash
python trace_rounds.py
```

**Configuration:** `PROMPT`, `K` (default 5), and `ROUNDS` (default 6) at the top of the file. The model pair comes from `speculative.py`.

**Output:**

- A console table per round, with token text, token ids, and a verdict for each guess: accepted, rejected (with the target's replacement), or discarded because it followed a rejection. Spaces display as `␣` and newlines as `↵`.
- `trace.html`, a color-coded version suitable for slides: green for accepted guesses, red for rejected, gray for discarded, blue for tokens from the target. It ends with the full generated text colored by which model produced each token.

If few guesses are accepted, try a more predictable prompt (code or lists), a smaller K, or a model pair with a smaller size gap.

---

## Adding a new program

Copy this template for each new script:

```markdown
## filename.py

**What it does:** one or two sentences.

**Usage:**

    python filename.py

**Configuration:** settings at the top of the file and their defaults.

**Output:** what it prints or writes, and where.
```

Also add a row to the **Programs** table near the top.

## Reference

Leviathan, Y., Kalman, M., and Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *Proceedings of the 40th International Conference on Machine Learning (ICML).*
