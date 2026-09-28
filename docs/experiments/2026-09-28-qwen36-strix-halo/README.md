# Qwen3.6-35B-A3B decode on Strix Halo (Radeon 8060S, RADV) — raw evidence

Branch `decode-gpu`; see the issue for the write-up.

- `bench.sh` — throughput runs (one warm-up discarded, 3 samples per arm), `results.tsv` columns:
  arm, prompt file, run (0 = warm-up), prompt tokens, TTFT s, decode tok/s = (generated − 1) / (wall − TTFT),
  MTP draft acceptance %.
- `llama-bench-tg.txt`, `llama-bench-pp.txt` — llama.cpp reference on the same machine (Q4_K_M, `-fa 1`, `-r 3`).
- `run_eval.py`, `build_items.py`, `items.jsonl` — judge-style quality check (15 JudgeBench pairs, MIT, 5 with a
  BIPIA-style injection); `eval-*.md` are its summaries (greedy: all configs one run; sampled: 3 seeds each).
