import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login; login()

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

#DRAFT_MODEL = "HuggingFaceTB/SmolLM-135M"
#TARGET_MODEL = "HuggingFaceTB/SmolLM-1.7B"

DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

PROMPT = "Explain how a binary search tree works."

MAX_NEW_TOKENS = 100

# Number of tokens proposed by the draft model at each iteration.
K = 5


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

print("Loading draft model...")

draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL,
    dtype=torch.float16
).to(device)

draft_model.eval()


print("Loading target model...")

target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    dtype=torch.float16
).to(device)

target_model.eval()


# ------------------------------------------------------------
# Encode prompt
# ------------------------------------------------------------

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(device)

print("\nPrompt:")
print(PROMPT)

print("\nInitial tokens:", input_ids.shape[1])


# ------------------------------------------------------------
# Speculative decoding
# ------------------------------------------------------------

generated_tokens = 0
accepted_tokens = 0
iterations = 0

start_time = time.perf_counter()


while generated_tokens < MAX_NEW_TOKENS:

    iterations += 1

    # ========================================================
    # STEP 1: Draft model proposes K tokens
    # ========================================================

    draft_start = time.perf_counter()

    draft_sequence = input_ids.clone()

    proposed_tokens = []

    with torch.no_grad():

        for _ in range(K):

            draft_output = draft_model(
                input_ids=draft_sequence
            )

            logits = draft_output.logits[:, -1, :]

            # Greedy decoding:
            next_token = torch.argmax(logits, dim=-1)

            proposed_tokens.append(next_token.item())

            draft_sequence = torch.cat(
                [draft_sequence, next_token.unsqueeze(0)],
                dim=1
            )

    draft_time = time.perf_counter() - draft_start


    # ========================================================
    # STEP 2: Target model verifies all proposed tokens
    # ========================================================

    target_start = time.perf_counter()

    # The target sees the original context plus
    # all tokens proposed by the draft model.

    proposed_tensor = torch.tensor(
        proposed_tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    target_sequence = torch.cat(
        [input_ids, proposed_tensor],
        dim=1
    )

    with torch.no_grad():

        target_output = target_model(
            input_ids=target_sequence
        )

    target_logits = target_output.logits


    # ========================================================
    # STEP 3: Compare draft predictions with target
    # ========================================================

    accepted_this_iteration = 0

    for i, proposed_token in enumerate(proposed_tokens):

        # Position corresponding to the proposed token.
        #
        # target_logits[:, input_position + i - 1, :]
        # predicts token i.

        position = input_ids.shape[1] + i - 1

        target_prediction = torch.argmax(
            target_logits[:, position, :],
            dim=-1
        ).item()

        if target_prediction == proposed_token:

            # Target agrees with draft.
            accepted_this_iteration += 1

        else:

            # Target disagrees.
            break


    # ========================================================
    # STEP 4: Add accepted tokens to our sequence
    # ========================================================

    if accepted_this_iteration > 0:

        accepted = proposed_tensor[
            :, :accepted_this_iteration
        ]

        input_ids = torch.cat(
            [input_ids, accepted],
            dim=1
        )

        accepted_tokens += accepted_this_iteration
        generated_tokens += accepted_this_iteration


    # --------------------------------------------------------
    # If a token was rejected, use the target model's token
    # --------------------------------------------------------

    if accepted_this_iteration < K:

        # Target's prediction at the first rejected position.

        position = input_ids.shape[1] - 1

        replacement_token = torch.argmax(
            target_logits[:, position, :],
            dim=-1
        )

        input_ids = torch.cat(
            [input_ids, replacement_token.unsqueeze(0)],
            dim=1
        )

        generated_tokens += 1


    target_time = time.perf_counter() - target_start


    # ========================================================
    # Display what happened
    # ========================================================

    print(
        f"Iteration {iterations:3d}: "
        f"proposed={K}, "
        f"accepted={accepted_this_iteration}, "
        f"draft={draft_time:.3f}s, "
        f"target={target_time:.3f}s"
    )


    # --------------------------------------------------------
    # Stop if we have generated enough
    # --------------------------------------------------------

    if generated_tokens >= MAX_NEW_TOKENS:
        break


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

total_time = time.perf_counter() - start_time

output_text = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(output_text)

print("\nGenerated tokens:", generated_tokens)
print("Accepted draft tokens:", accepted_tokens)

if generated_tokens > 0:
    print(
        "Acceptance rate:",
        f"{accepted_tokens / generated_tokens:.2%}"
    )

print("Iterations:", iterations)
print("Time:", f"{total_time:.3f} seconds")
print(
    "Tokens/sec:",
    f"{generated_tokens / total_time:.2f}"
)
