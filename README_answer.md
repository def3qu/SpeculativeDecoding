# Speculative Answer

A chat program that answers your prompts with speculative decoding. A small
draft model (Qwen2.5 0.5B) and a large target model (Qwen2.5 7B) work together.
Nothing is measured, and the large model is never run on its own.

## How it works (K = 3)

1. The draft model quickly guesses the next 3 tokens.
2. The target model checks all 3 guesses in a single pass.
3. Guesses that match what the target would have written are kept, and the
   target adds one token of its own. Each round yields 1 to 4 tokens.

The answer is exactly what the 7B model would write alone (greedy decoding,
always the most likely token). Only the speed is different.

## Setup

Python 3.9 or newer and about 16 GB of free memory.

```
pip install torch transformers
python speculative_answer.py
```

The first run downloads both models. No account is needed. A GPU (NVIDIA or
Apple Silicon) is strongly recommended.

For a smaller computer, pick a smaller target from the same family:

```
python speculative_answer.py --target Qwen/Qwen2.5-3B-Instruct
```

The draft and target must come from the same family so they share a tokenizer.

## Commands

```
:paste            multi-line prompt, finish with a line containing only END
:new              start a fresh conversation
:examples         list sample prompts
:example <n>      run sample prompt n
:max <n>          maximum new tokens
:help
:quit
```

Options: `--draft`, `--target`, `--max-new`, `--device`, `--dtype`,
`--prompt "..."`, `--prompt-file file.txt`.