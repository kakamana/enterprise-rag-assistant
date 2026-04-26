# Model Card — Enterprise RAG Assistant

## Intended use
Decision-aid for employees and Compliance staff: surface the cited policy paragraph(s) for any natural-language question, with a faithfulness score so the user can decide whether to act. Never a substitute for the policy owner.

## Training data
Synthetic 12-domain policy corpus (5,000 paragraphs) + 1,000 Q&A pairs with ground-truth source IDs (see `data/data_card.md`). The reranker LR is trained on 80% of the Q&A; recall is reported on the held-out 20%.

## Model family
- **Sparse retrieval:** rank_bm25 Okapi BM25 (k1=1.5, b=0.75)
- **Dense retrieval (notebook fallback):** TF-IDF (1-2 grams) + TruncatedSVD(256) + L2 normalize
- **Hybrid blend:** alpha-weighted min-max normalized scores
- **Reranker:** sklearn LogisticRegression on `[bm25_score, dense_score, ranks, length_ratio, domain_match]`
- **Faithfulness:** cosine similarity between the answer encoding and the mean retrieved-source encoding

## Production swap-in
- Dense retrieval -> sentence-transformers `bge-small-en` or `e5-small-v2`
- Reranker -> cross-encoder `ms-marco-MiniLM-L-6-v2`
- Answer composition -> any hosted endpoint (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) prompted with the retrieved passages

## Metrics (held-out test, to be filled)
| Metric | Target |
|--------|--------|
| Recall@5 | >= 0.85 |
| MRR@10 | >= 0.70 |
| Faithfulness | >= 0.80 |
| Citation precision (manual 50-q audit) | >= 0.75 |
| p95 latency | < 300 ms (CPU) |

## Limitations
- Synthetic templates only; real policy registers have idiomatic clauses the model has never seen.
- No multi-hop reasoning across > 2 paragraphs.
- Faithfulness is a **proxy** (cosine similarity), not a ground-truth NLI verdict.

## Ethical considerations
- Decision-aid disclaimer on every response.
- Abstain if faithfulness < 0.50, surfacing "escalate to policy owner" instead of guessing.
- All retrieval runs on-prem; the optional generator is configurable.

## Retraining
- Index rebuild on every policy publication (CI job).
- Reranker retrained monthly or when recall@5 drops > 5 pts.

## Ownership
- On-call DS: Asad
- Runbook: `mlops/runbook.md` (TBD)
