# How alphaTest.py Uses PyTorch

Roughly in the order the work happens. Line numbers refer to alphaTest.py.

## 1. Choosing where the math runs (lines 144–148)

`torch.device("mps")` selects Apple's GPU through Metal Performance Shaders. On the Mac Studio, every forward pass runs there. Because Apple Silicon has unified memory, "moving to the GPU" places tensors where the GPU can use them rather than copying them across a bus as on a PC with a discrete card.

## 2. The models are PyTorch objects (line 152)

`AutoModelForCausalLM.from_pretrained` builds a PyTorch `nn.Module`, a network whose weights are tensors. Hugging Face supplies the architecture and weights; PyTorch does all the computing.

- `dtype=torch.float16` stores each weight in 2 bytes, so the 8B target needs about 16 GB instead of 32 GB. Halving the bytes also roughly halves the time spent reading weights, which is the bottleneck the talk is about.
- `.to(device)` puts the weights on the GPU.
- `.eval()` switches off training-only behavior.

## 3. Tokens are tensors

Text becomes a tensor of integer token IDs with shape `[1, sequence_length]`. The leading 1 is the batch dimension: one sequence at a time, the single-user case.

- `torch.tensor([[token]])` (line 237) wraps one new token in that shape.
- `torch.cat(..., dim=1)` (lines 325–326) appends the generated tokens to the prompt along the sequence dimension.

## 4. A forward pass returns logits

Calling `model(input_ids=...)` runs the whole network and returns logits with shape `[1, sequence_length, vocab_size]`: one score per vocabulary word, at every position. Slicing picks out what is needed:

- `logits[0, -1, :vocab]` is the last position only: the prediction for the next token (generation).
- `logits[0, prompt_len - 1:-1, :vocab]` is every position whose prediction is a generated token. The `- 1` shift exists because position *i* predicts token *i* + 1.

The `:vocab` trims padded columns: the models' output layers can be wider than the real vocabulary (Qwen2.5 0.5B and 7B are padded to different widths).

## 5. Inference mode and the KV cache

- `@torch.no_grad()` (lines 214, 251) tells PyTorch not to record operations for backpropagation. Nothing is trained, so this saves memory and time.
- `use_cache=True` and `past_key_values` keep each layer's attention keys and values from earlier tokens as tensors, so each generation step processes only the one new token instead of the whole sequence.

## 6. The alpha math is tensor math

After `.float().cpu()`, the logits are converted to 32-bit floats on the CPU. Summing 128K to 152K probabilities in float16 would lose precision, and these are small tensors now, so the CPU is fine.

- `torch.softmax(logits / temperature)` (line 200) turns scores into probabilities. Dividing by the temperature first sharpens (T < 1) or flattens (T > 1) the distribution.
- `torch.zeros_like` plus `scatter_` (line 197) builds the one-hot distribution for temperature 0.
- `torch.multinomial(probs, 1, generator=gen)` (line 230) samples a token. The seeded `torch.Generator` (line 316) makes every run reproducible.
- **Line 337 is the heart of the experiment:**

  ```python
  betas = torch.minimum(p, q).sum(dim=-1)
  ```

  `p` and `q` both have shape `[n_tokens, vocab]`. `torch.minimum` takes the elementwise minimum, and `.sum(dim=-1)` adds across the vocabulary, giving one β per position in a single vectorized step with no Python loop over 128K entries.
- `torch.isfinite` (line 187) catches float16 overflow before it corrupts the results.

## 7. Crossing back into Python

`.item()` and `int(...)` pull a single number out of a tensor so ordinary Python can use it, for example to test `token in eos_ids`. Each call makes the CPU wait for the GPU to finish, which is one reason generation loops are written to keep work on tensors as long as possible.

## Summary

PyTorch holds the weights, runs both networks, and does the probability math. Hugging Face only supplies the model definitions, the weights, and the tokenizer.