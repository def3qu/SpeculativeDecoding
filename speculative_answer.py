#!/usr/bin/env python3
"""
speculative_answer.py

A chat program that answers your prompts with speculative decoding, using a
small draft model and a large target model. Nothing is measured and the target
model is never run on its own: each answer is produced by the two models
working together.

How it works (draft length K = 3):
  1. The small draft model (Qwen2.5 0.5B) quickly guesses the next 3 tokens.
  2. The large target model (Qwen2.5 7B) checks all 3 guesses in a single pass.
  3. Guesses that match what the target would have written are kept, and the
     target adds one token of its own. Each round yields 1 to 4 tokens.

The answer is exactly what the large model would write by itself (greedy
decoding, always the most likely token). Only the speed is different.

Setup (Python 3.9 or newer, about 16 GB of free memory):

    pip install torch transformers

    python speculative_answer.py
    python speculative_answer.py --prompt "Explain how a hash table works."
    python speculative_answer.py --prompt-file my_text.txt

For a smaller computer, use a smaller pair from the same family, for example:

    python speculative_answer.py --draft Qwen/Qwen2.5-0.5B-Instruct --target Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("This program needs PyTorch and Transformers. Install them with:\n"
             "    pip install torch transformers")


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEFAULT_DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW = 512
K = 3                       # tokens the draft model guesses per round

EXAMPLES = [
    "Explain how a hash table works in a few short paragraphs.",
    "Write a Python function that checks whether a string is a palindrome, with a short explanation.",
    "What is the difference between a process and a thread?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
]


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
    # A 7B model needs about 15 GB in a 16-bit format and 30 GB in float32,
    # so even a CPU gets a 16-bit format by default.
    return torch.float16 if device.type in ("cuda", "mps") else torch.bfloat16


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
    Used to throw away the entries for guessed tokens the model rejected."""
    remove = cache.get_seq_length() - keep
    if remove > 0:
        cache.crop(-remove)


# ------------------------------------------------------------
# Speculative decoding
# ------------------------------------------------------------

@torch.no_grad()
def generate(draft, target, input_ids, eos_ids, vocab, max_new, device, on_tokens=None):
    """Greedy speculative decoding (Leviathan et al., 2023, temperature 0).

    Each round:
      1. The draft model proposes K tokens, one forward pass each.
      2. The target scores all K proposals in ONE forward pass. Its K + 1
         output rows give its own choice at each proposal position, plus one
         extra position after the last proposal.
      3. Keep the longest prefix where draft and target agree, then add the
         target's own token at the first disagreement (or the extra token if
         all K were right). Every round yields between 1 and K + 1 tokens.
      4. Roll both KV caches back to the tokens that were kept.
    """
    seq = input_ids[0].tolist()
    new_tokens = []

    # Process every prompt token except the last. The first round feeds the
    # last prompt token, so every new token comes from the loop below.
    d_cache = draft(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
    t_cache = target(input_ids=input_ids[:, :-1], use_cache=True).past_key_values

    done = False
    while not done:
        base = len(seq)

        # Step 1: the draft model proposes K tokens.
        draft_input = seq[d_cache.get_seq_length():]
        proposed = []
        for _ in range(K):
            out = draft(
                input_ids=torch.tensor([draft_input], device=device),
                past_key_values=d_cache,
                use_cache=True,
            )
            d_cache = out.past_key_values
            token = int(out.logits[0, -1, :vocab].argmax())
            proposed.append(token)
            draft_input = [token]

        # Step 2: the target checks all K proposals in one forward pass.
        verify_input = seq[t_cache.get_seq_length():] + proposed
        out = target(
            input_ids=torch.tensor([verify_input], device=device),
            past_key_values=t_cache,
            use_cache=True,
        )
        t_cache = out.past_key_values
        target_preds = out.logits[0, -(K + 1):, :vocab].argmax(dim=-1).tolist()

        # Step 3: accept the longest agreeing prefix, then the target's token.
        accepted = 0
        while accepted < K and proposed[accepted] == target_preds[accepted]:
            accepted += 1

        for token in proposed[:accepted] + [target_preds[accepted]]:
            seq.append(token)
            new_tokens.append(token)
            if token in eos_ids or len(new_tokens) >= max_new:
                done = True
                break

        # Step 4: drop cache entries for rejected guesses.
        crop_cache(d_cache, base + accepted)
        crop_cache(t_cache, base + accepted)

        if on_tokens:
            on_tokens(new_tokens, done)

    return new_tokens


# ------------------------------------------------------------
# Chat
# ------------------------------------------------------------

class Chat:
    def __init__(self, draft_name, target_name, device, dtype):
        self.device = device
        print("Loading", target_name, "(target) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(target_name)
        self.target = load_model(target_name, device, dtype)
        print("Loading", draft_name, "(draft) ...", flush=True)
        self.draft = load_model(draft_name, device, dtype)
        self.vocab = len(self.tokenizer)   # ignore padding columns in the logits
        self.eos_ids = get_eos_ids(self.tokenizer, self.target)
        self.messages = []

    def encode(self):
        encoded = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"].to(self.device)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def reset(self):
        self.messages = []

    def ask(self, prompt, max_new, show_text=True):
        """Answer one prompt in the context of the conversation so far."""
        self.messages.append({"role": "user", "content": prompt})
        input_ids = self.encode()

        printed = 0

        def stream(tokens, done):
            nonlocal printed
            text = self.decode(tokens)
            if not done and text.endswith("\ufffd"):
                return  # wait for the rest of a character split across tokens
            if show_text:
                print(text[printed:], end="", flush=True)
            printed = len(text)

        tokens = generate(self.draft, self.target, input_ids, self.eos_ids,
                          self.vocab, max_new, self.device, stream)
        answer = self.decode(tokens)
        self.messages.append({"role": "assistant", "content": answer})
        print()
        return answer

    def warm_up(self):
        self.ask("Say hello.", 8, show_text=False)
        self.reset()


# ------------------------------------------------------------
# Command line and interactive loop
# ------------------------------------------------------------

HELP = """
Type a prompt and press Enter. Follow-up questions remember the conversation.
Commands:
  :paste            enter a long or multi-line prompt; finish with a line containing only END
  :new              start a fresh conversation
  :examples         list sample prompts
  :example <n>      run sample prompt number n
  :max <n>          maximum new tokens per answer
  :help             show this list
  :quit             exit
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Answer prompts with speculative decoding (draft length K = 3).")
    parser.add_argument("--draft", default=DEFAULT_DRAFT,
                        help=f"draft model (default: {DEFAULT_DRAFT})")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"target model (default: {DEFAULT_TARGET})")
    parser.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW,
                        help=f"maximum new tokens per answer (default: {DEFAULT_MAX_NEW})")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="where to run the models (default: auto)")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="number format (default: float16 on a GPU, bfloat16 on a CPU)")
    parser.add_argument("--prompt", help="answer this prompt once and exit")
    parser.add_argument("--prompt-file", help="answer the text in this file once and exit")
    args = parser.parse_args()
    if args.max_new < 1:
        parser.error("--max-new must be at least 1")
    return args


def read_paste():
    print("Paste your text, then type END on a line by itself:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(f"Device: {device.type}, number format: {str(dtype).replace('torch.', '')}")
    if device.type == "cpu":
        print("No GPU found, so this will be slow. Try a smaller target, for example "
              "--target Qwen/Qwen2.5-1.5B-Instruct.")

    try:
        chat = Chat(args.draft, args.target, device, dtype)
    except Exception as error:  # download and memory problems end up here
        message = str(error)
        print(f"\nCould not load the models: {message[:300]}")
        if "memory" in message.lower():
            print("Not enough memory. Try a smaller target, for example "
                  "--target Qwen/Qwen2.5-3B-Instruct.")
        sys.exit(1)

    print("Warming up...", flush=True)
    chat.warm_up()

    max_new = args.max_new

    if args.prompt or args.prompt_file:
        if args.prompt_file:
            with open(args.prompt_file, encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = args.prompt
        chat.ask(prompt, max_new)
        return

    print(f"\nReady. Draft: {args.draft}. Target: {args.target}. K = {K}.")
    print(HELP)

    while True:
        try:
            line = input("\nprompt> ").strip()
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
                elif command == "new":
                    chat.reset()
                    print("Started a new conversation.")
                elif command == "paste":
                    text = read_paste()
                    if text:
                        chat.ask(text, max_new)
                elif command == "max" and argument.isdigit() and int(argument) >= 1:
                    max_new = int(argument)
                elif command == "examples":
                    print()
                    for number, text in enumerate(EXAMPLES, start=1):
                        print(f"  {number}. {text}")
                elif command == "example" and argument.isdigit() and 1 <= int(argument) <= len(EXAMPLES):
                    text = EXAMPLES[int(argument) - 1]
                    print(f"Prompt: {text}\n")
                    chat.ask(text, max_new)
                else:
                    print("Unrecognized command.")
                    print(HELP)
                continue

            chat.ask(line, max_new)
        except KeyboardInterrupt:
            print("\nInterrupted.")

    print("Goodbye.")


if __name__ == "__main__":
    main()
