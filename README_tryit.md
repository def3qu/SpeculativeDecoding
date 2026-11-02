## Try it yourself

Three programs let you run speculative decoding on your own computer. They need
Python 3.9 or newer and two packages:

```
pip install torch transformers
```

The first run downloads the models. A GPU (NVIDIA, or Apple Silicon) is strongly
recommended. All three use greedy decoding, so the answer is exactly what the
large model would write alone; only the speed changes.

| Program | What it does | Start here if you want to |
|---|---|---|
| `speculative_playground.py` | Answers a prompt with the large model alone and again with speculative decoding, then prints the measured acceptance rate (alpha), cost ratio (c), and the speedup the formula predicts. | See the speedup and check the theory on your machine |
| `speculative_answer.py` | A chat that answers with speculative decoding only (draft length K = 3). Nothing is measured. | Just use it |


```
python speculative_playground.py                 # default pair, needs about 8 GB of memory
python speculative_playground.py --pair tiny     # about 4 GB, for small machines
python speculative_answer.py                     # Qwen2.5 0.5B drafts for Qwen2.5 7B, about 16 GB
python speculative_answer.py --target Qwen/Qwen2.5-3B-Instruct   # smaller machine
```

In the playground, type `:sweep` to see the speedup for K = 1, 2, 3, 5 and 8 in
one run, and `:examples` for sample prompts. Each program has more detail in
its own README (`README_playground.md`, `README_answer.md`).

The draft and target models must come from the same family so they share a
tokenizer.

Note: these programs use a KV cache for both models. The research benchmark
(`speculativeTestSpec.py`) does not, so absolute speeds differ between them.
