# Spec-Bench prompt set

`spec_bench_questions.jsonl` is the question file from Spec-Bench, the standard benchmark for speculative decoding (Xia et al., "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding," Findings of ACL 2024).

- Source: https://github.com/hemingkx/Spec-Bench, file `data/spec_bench/question.jsonl`
- Commit: fd2c1cd7d2201ef71db4c5f4e455008f017967bf (Apr 22, 2025), downloaded Oct 13, 2026
- License: the repository is Apache 2.0; the questions come from the public datasets listed below

## Format

One JSON object per line with three fields:

```json
{"question_id": 161, "category": "translation", "turns": ["Translate German to English: ..."]}
```

`turns` is a list. 400 questions have one turn; the 80 MT-Bench questions have two (a follow-up question asked after the model answers the first).

## Contents (480 questions, 6 tasks of 80)

| Task | Categories in the file | IDs | Source dataset |
|---|---|---|---|
| Multi-turn conversation | writing, roleplay, reasoning, math, coding, extraction, stem, humanities (10 each) | 81–160 | MT-Bench |
| Translation | translation | 161–240 | WMT14 German to English |
| Summarization | summarization | 241–320 | CNN/DailyMail |
| Question answering | qa | 321–400 | Natural Questions |
| Mathematical reasoning | math_reasoning | 401–480 | GSM8K |
| Retrieval-augmented generation | rag | 481–560 | Natural Questions with retrieved passages |

## Notes for use with alphaTest.py

- For alpha measurements, use the first turn of each question. Using the second MT-Bench turn would require generating the first answer and putting it in the chat history.
- Report the multi-turn task as one group of 80, as the Spec-Bench papers do, or keep its 8 subcategories (only 10 prompts each).
- Summarization and RAG prompts are long (full news articles and retrieved passages), so they take longer per sample and may show a different cost ratio, as the summarization category did in the Oct 12 benchmark.