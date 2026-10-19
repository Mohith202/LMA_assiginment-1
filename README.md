# RAG: Baseline retrieval vs. Cross-Encoder re-ranking

Everything is in one notebook: [rag_baseline_vs_rerank.ipynb](rag_baseline_vs_rerank.ipynb) (already executed, outputs included).

## What was done
- **Document:** NVIDIA fiscal-2024 Annual Report / 10-K / proxy statement, 187 pages (`data/nvidia_annual_report_fy2024.pdf`). Split into 700 overlapping chunks (180 words, 40 overlap).
- **Pipeline 1 (baseline):** `BAAI/bge-small-en-v1.5` bi-encoder, FAISS cosine search, top-10 retrieved, top-3 sent to the LLM (Answer 1).
- **Pipeline 2 (re-ranking):** the same top-10 scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, best 3 sent to the LLM with the identical prompt (Answer 2).
- **LLM:** `qwen2.5:7b-instruct` via Ollama, temperature 0, fixed seed.
- **Failure-mode queries:** 9 hand-written queries covering exceptions, negation, a keyword trap, buried chunks, two controls and two negative results. For each, the rank of the evidence chunk before/after re-ranking and both answers are shown.
- **Synthetic benchmark:** 260 LLM-written questions (paraphrase / negation / exception / factual) with the source chunk as gold label, to check the effect beyond hand-picked queries. Questions are cached in `data/synthetic_questions.json`.
- **Latency:** retrieval vs. retrieval + re-ranking vs. LLM generation, after warm-up. Also the cost of running the cross-encoder over the whole corpus, extrapolated.

## Main numbers (from the executed notebook)
| | Baseline | Re-ranked |
|---|---|---|
| Evidence chunk in Top-3 (260 synthetic questions) | 71.5% | 83.5% |
| MRR | 0.60 | 0.74 |
| Retrieval latency (mean) | 5.8 ms | 14.7 ms |
| LLM generation (mean, same for both) | ~490 ms | ~490 ms |

34 questions had their evidence promoted into the Top-3, 3 were pushed out, and 31 were never in the Top-10 (re-ranking cannot help those).

## Things that did not work or need care
- Evidence for "What is NOT allowed under the insider trading policy?" and "Which directors are not independent?" never reaches the baseline Top-10 (dense ranks #110 and #51), so re-ranking the Top-10 cannot fix them. Re-ranking only reorders what retrieval already found.
- For the "Fall season" query the evidence moved to rank 1, but both answers still said 2023 (instead of 2022) because another top-3 chunk mentions a different year.
- The 10-K states some facts twice (MD&A and notes), and the synthetic benchmark counts a hit on the duplicate as a miss. Both pipelines are affected equally.
- The correctness check in the notebook is a rough regex check (a refusal never counts as correct); read the printed answers too.
- Questions in the synthetic benchmark are LLM-written, so some are awkward.

## Run it
```
pip install sentence-transformers pypdf faiss-cpu ollama pandas jupyter
ollama pull qwen2.5:7b-instruct
jupyter nbconvert --to notebook --execute --inplace rag_baseline_vs_rerank.ipynb
```
The PDF is downloaded automatically if `data/nvidia_annual_report_fy2024.pdf` is missing. Output files: `results_rank_shifts.csv`, `results_latency.csv`, `results_synthetic_benchmark.csv`, `results_answers.json`.

The written report (architecture comparison, rank-shift discussion, latency table commentary, scaling question) is a separate deliverable and is not part of this repository.
