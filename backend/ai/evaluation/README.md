# Local LLM Evaluation (Grammar / Summarization / Rewrite)

This folder provides a lightweight evaluation harness for your local Transformers + PEFT/LoRA LLM.

## What metrics are produced

For each task, the benchmark writes:

- `backend/ai/evaluation/outputs/<task>_summary.json`
- `backend/ai/evaluation/outputs/<task>_results.jsonl` (per-example details)

Metrics:

- `grammar correction accuracy`
  - `grammar_accuracy`: normalized edit similarity between prediction and reference
- `summarization quality`
  - `rouge_1_f1`
  - `rouge_l_f1`
  - `summary_similarity`
- `rewrite quality`
  - `rewrite_rouge_l_f1`
  - `rewrite_similarity`
- `response latency`
  - `avg_latency_ms`
  - `latency_percentiles_ms` (`p50`, `p90`, `p95`)

Scoring is intentionally lightweight (no external metric packages) so it runs in your current backend environment.

## Run

Benchmarks use your existing `backend/services/local_llm.py`.
Make sure your model env vars are set (for base + optional adapter), then run:

```bash
python backend/ai/evaluation/benchmark_grammar.py --limit 50 --max-new-tokens 192
python backend/ai/evaluation/benchmark_summarization.py --limit 20 --max-new-tokens 192
python backend/ai/evaluation/benchmark_rewrite.py --limit 50 --max-new-tokens 192
python backend/ai/evaluation/run_all.py --limit 25 --max-new-tokens 192
```

## Outputs

The `*_summary.json` file includes aggregated scores + latency for that task.

