import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM-1.7B"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)

prompt = "Explain how a binary search tree works. "

inputs = tokenizer(prompt, return_tensors="pt").to(device)

start = time.perf_counter()

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

elapsed = time.perf_counter() - start

new_tokens = output.shape[1] - inputs["input_ids"].shape[1]

print(tokenizer.decode(output[0], skip_special_tokens=True))

print()
print("Generated tokens:", new_tokens)
print("Time:", elapsed)
print("Tokens/sec:", new_tokens/elapsed)
