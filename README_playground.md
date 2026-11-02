# Speculative Decoding Playground

A small program for trying speculative decoding on your own computer. Type a
prompt and it generates the answer twice: once with a large model alone, and
once with a small "draft" model guessing ahead while the large model checks
the guesses. The answers are identical; only the speed differs.

## Setup

You need Python 3.9 or newer and about 8 GB of free memory for the default
models.

```
pip install torch transformers
python speculative_playground.py
```

The first run downloads the models, which can take a few minutes. The default
models need no account. A GPU (NVIDIA, or Apple Silicon) is strongly
recommended; on a CPU alone, use `--pair tiny` and a small `--max-new`.

## Things to try

| Try this | What you learn |
|---|---|
| Type any prompt | How much faster speculative decoding is on your machine |
| `:show on`, then a prompt | Which guesses the large model accepts and rejects |
| `:k 1`, `:k 3`, `:k 8` | Longer drafts help, then hurt |
| `:sweep` | The whole speedup-versus-K curve in one run |
| `:examples`, `:example 3` | Predictable text (math, code) speeds up more than open-ended text (translation, stories) |
| `:mode speculative` | Speculative decoding alone, with the acceptance statistics |

After each comparison the program prints the measured acceptance rate (alpha),
the cost ratio of the two models (c), and the speedup the published formula
predicts, so you can check the theory against your own hardware.

## Commands

```
:mode compare | baseline | speculative
:k <n>            draft length
:max <n>          maximum new tokens
:show on | off    show every round of guesses
:sweep [prompt]   try K = 1, 2, 3, 5, 8
:examples         list sample prompts
:example <n>      run a sample prompt
:help
:quit
```

## Options

```
python speculative_playground.py --pair tiny      # about 4 GB of memory
python speculative_playground.py --pair small     # about 8 GB (default)
python speculative_playground.py --pair qwen      # about 16 GB
python speculative_playground.py --pair llama     # needs a Hugging Face login
python speculative_playground.py --prompt "Explain how a hash table works."
python speculative_playground.py --draft ORG/SMALL --target ORG/LARGE
```

A custom draft and target must come from the same model family, because they
have to share a tokenizer.

## Notes

* Decoding is greedy (always the most likely token), as in the research
  experiments. The two answers should match exactly. Very rarely they differ
  slightly, because half-precision math rounds a little differently when
  several tokens are checked at once.
* The speedup depends on how much smaller the draft model is. If the two models
  are close in size, the draft is not much cheaper and the gain is small.
* Timings are for this simple implementation, not a production inference
  engine, so absolute speeds will be lower than in tools like Ollama. The
  baseline and speculative runs use the same implementation, so the speedup
  compares like with like.