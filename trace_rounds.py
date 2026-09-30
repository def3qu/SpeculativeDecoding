"""
trace_rounds.py

Runs a few rounds of speculative decoding and shows every token:
what the draft model guessed, what the target model wanted at the same
spot, and what was kept. Prints a table per round and writes trace.html,
a color-coded version for slides.

Uses the models, step(), and encode() from speculative.py, so the model
pair is whatever speculative.py is configured with.
"""

import html

import torch
from transformers import DynamicCache

import speculative as sd

PROMPT = "Explain how a binary search tree works."
K = 5
ROUNDS = 6
OUT_HTML = "trace.html"


def show(tok_id):
    """Token text with visible spaces and newlines."""
    return sd.tokenizer.decode([tok_id]).replace(" ", "␣").replace("\n", "↵")


# ------------------------------------------------------------
# Run the rounds, recording everything
# ------------------------------------------------------------

@torch.no_grad()
def trace():
    draft_cache, target_cache = DynamicCache(), DynamicCache()
    seq = sd.encode(PROMPT)
    prompt_len = seq.shape[1]

    sd.step(sd.draft_model, draft_cache, seq[:, :-1])
    sd.step(sd.target_model, target_cache, seq[:, :-1])

    rounds = []
    for n in range(1, ROUNDS + 1):
        # 1. Small model guesses K tokens, one at a time.
        draft_seq = seq
        for _ in range(K):
            tok = sd.step(sd.draft_model, draft_cache, draft_seq)[:, -1].argmax(-1, keepdim=True)
            draft_seq = torch.cat([draft_seq, tok], dim=1)
        drafts = draft_seq[0, seq.shape[1]:].tolist()

        # 2. Large model reads the text plus all K guesses in ONE pass.
        logits = sd.step(sd.target_model, target_cache, draft_seq)
        target = logits[0, -(K + 1):].argmax(-1).tolist()

        # 3. Keep guesses until the first disagreement.
        a = 0
        while a < K and drafts[a] == target[a]:
            a += 1
        kept = drafts[:a] + [target[a]]    # plus the target's fix, or the bonus

        rd = {"n": n, "context": seq[0, prompt_len:].tolist(),
              "drafts": drafts, "target": target, "a": a, "kept": kept}
        rounds.append(rd)
        print_round(rd)

        seq = torch.cat([seq, torch.tensor([kept], device=sd.device)], dim=1)
        keep = seq.shape[1] - 1
        target_cache.crop(keep)
        draft_cache.crop(keep)

        if any(t in sd.EOS_IDS for t in kept):
            break

    return rounds


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

def print_round(rd):
    print(f"\n=== Round {rd['n']} ===")
    tail = sd.tokenizer.decode(rd["context"][-12:]) if rd["context"] else "(start of answer)"
    print("Text so far ends with:", repr(tail))
    print(f"{'#':>2}  {'draft guessed':<16}{'id':>7}   {'target wanted':<16}{'id':>7}   result")

    for i in range(K):
        d, t = rd["drafts"][i], rd["target"][i]
        if i < rd["a"]:
            result = "accepted"
        elif i == rd["a"]:
            result = f"REJECTED, keep target's {show(t)!r}"
        else:
            result = "discarded (comes after a rejection)"
        print(f"{i + 1:>2}  {show(d):<16}{d:>7}   {show(t):<16}{t:>7}   {result}")

    if rd["a"] == K:
        b = rd["target"][K]
        print(f"    all {K} accepted, bonus token {show(b)!r} ({b}) from the same target pass")

    print(f"Cost: {K} draft passes + 1 target pass -> {len(rd['kept'])} tokens kept: "
          f"{sd.tokenizer.decode(rd['kept'])!r}")


# ------------------------------------------------------------
# HTML output for slides
# ------------------------------------------------------------

def chip(tok, cls):
    return f'<span class="tok {cls}">{html.escape(show(tok))}<sub>{tok}</sub></span>'


def write_html(rounds):
    blocks = []
    for rd in rounds:
        a = rd["a"]
        guesses = "".join(
            chip(d, "acc" if i < a else "rej" if i == a else "dis")
            for i, d in enumerate(rd["drafts"]))
        fix = rd["target"][a]
        fix_label = "bonus token" if a == K else "target's fix"
        kept = "".join(chip(t, "acc") for t in rd["drafts"][:a]) + chip(fix, "tgt")
        blocks.append(f"""
<div class="round">
  <h2>Round {rd['n']}</h2>
  <div class="line"><span class="lbl">Small model guessed</span>{guesses}</div>
  <div class="line"><span class="lbl">Large model checked</span>
    {a} of {K} match; {fix_label}: {chip(fix, 'tgt')}</div>
  <div class="line"><span class="lbl">Kept</span>{kept}
    <span class="cost">{K} small passes + 1 large pass &rarr; {len(rd['kept'])} tokens</span></div>
</div>""")

    # Whole answer, colored by who produced each token.
    answer = []
    for rd in rounds:
        answer += [chip(t, "acc") for t in rd["drafts"][:rd["a"]]]
        answer.append(chip(rd["target"][rd["a"]], "tgt"))

    n_tokens = sum(len(rd["kept"]) for rd in rounds)
    n_rounds = len(rounds)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Speculative decoding trace</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  .sub {{ color: #666; margin-bottom: 24px; }}
  .round {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }}
  h2 {{ font-size: 18px; margin: 0 0 8px; }}
  .line {{ margin: 6px 0; line-height: 2.2; }}
  .lbl {{ display: inline-block; width: 170px; color: #555; }}
  .tok {{ font-family: Menlo, monospace; font-size: 16px; padding: 3px 7px; margin: 2px;
          border-radius: 5px; border: 1px solid transparent; }}
  .tok sub {{ font-size: 10px; color: #777; margin-left: 3px; }}
  .acc {{ background: #d9f2dc; border-color: #5bb36a; }}
  .rej {{ background: #fbdcdc; border-color: #d65c5c; text-decoration: line-through; }}
  .dis {{ background: #eee; color: #999; text-decoration: line-through; }}
  .tgt {{ background: #dce8fb; border-color: #4f7fd6; }}
  .cost {{ color: #666; margin-left: 12px; font-size: 14px; }}
  .legend span {{ margin-right: 14px; }}
  .answer {{ line-height: 2.4; }}
</style></head><body>
<h1>Speculative decoding, token by token</h1>
<div class="sub">Draft: {html.escape(sd.DRAFT_MODEL)} &nbsp;|&nbsp; Target: {html.escape(sd.TARGET_MODEL)}
 &nbsp;|&nbsp; K = {K} &nbsp;|&nbsp; Prompt: &ldquo;{html.escape(PROMPT)}&rdquo;</div>
<p class="legend">
  <span class="tok acc">accepted guess</span>
  <span class="tok rej">rejected guess</span>
  <span class="tok dis">discarded</span>
  <span class="tok tgt">from large model</span>
  &nbsp; Small numbers are token ids.</p>
{''.join(blocks)}
<h2>The answer so far</h2>
<p class="sub">{n_tokens} tokens from {n_rounds} large-model passes
 (the large model alone would need {n_tokens}).</p>
<div class="answer">{''.join(answer)}</div>
</body></html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\nWrote {OUT_HTML}: {n_tokens} tokens from {n_rounds} large-model passes.")


if __name__ == "__main__":
    write_html(trace())
