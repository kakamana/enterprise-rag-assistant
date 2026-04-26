# Data Sources — #19 Enterprise RAG Assistant

## Primary

| # | Source | URL | Fields used | License |
|---|--------|-----|-------------|---------|
| 1 | Synthetic policy corpus | `src/enterprise_rag/data.py::build_corpus` | All 4 cols (see data_card.md) | MIT |
| 2 | Synthetic Q&A | `src/enterprise_rag/data.py::build_qa` | All 4 cols | MIT |

## Reference / inspiration

| Source | URL | Use |
|--------|-----|-----|
| RAGAS | https://github.com/explodinggradients/ragas | Faithfulness metric design |
| BEIR benchmark | https://github.com/beir-cellar/beir | Retrieval evaluation conventions (Recall@k, MRR) |
| MS MARCO | https://microsoft.github.io/msmarco/ | Cross-encoder reranker design (production swap-in) |

## Optional real-corpus override
Drop `*.txt` files into `data/raw/policies/<domain>/` — `enterprise_rag.data` will detect and ingest them in place of the synthetic corpus. This is **opt-in** and not used in the default reproducible flow.

## How to (re)build
```bash
python -m enterprise_rag.data        # 5,000 paragraphs + 1,000 Q&A
python -m enterprise_rag.models      # BM25 + TF-IDF/SVD + LR reranker artifacts
```

## Attribution
Synthetic content; no third-party data is included.
