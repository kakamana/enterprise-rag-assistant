# Enterprise RAG Assistant over Policy Documents

> **Ask company policy in plain English; get a grounded, cited answer with a faithfulness score — not a hallucination.** Hybrid retrieval (BM25 + dense embeddings) with cross-encoder reranking and a RAGAS-style faithfulness gate, exposed via FastAPI + a Next.js chat cockpit.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- Most internal "ask the policy" tools answer fluently and cite *nothing*. This one returns the answer **plus** the source paragraphs **plus** a faithfulness score so a Compliance lead can see at a glance whether to trust the response.
- Built for a realistic **12-domain enterprise policy corpus** (HR, Finance, IT, Travel, Procurement, Compliance, Security, Legal, Health & Safety, Sustainability, Data, Vendor) with 5,000 paragraphs and 1,000 evaluation Q&A pairs that carry ground-truth citations.
- The notebook stack is fully **Dataiku DSS compatible** — sklearn / TF-IDF / BM25 only, no external API calls.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — Hybrid retrieval, reranking, faithfulness](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) · [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) · [Source](./src/enterprise_rag/) · [API](./api/main.py) · [UI](./ui/app/page.tsx)
- [CLAUDE.md](./CLAUDE.md) — paste prompt to resume in this folder

## Headline results (target)

| Metric | Baseline (BM25 only) | Hybrid + Rerank | Target |
|---|---|---|---|
| Recall@5 | 0.61 | **0.86** | +25 pts |
| MRR@10 | 0.42 | **0.71** | +29 pts |
| Faithfulness (cosine answer↔source) | 0.55 | **0.82** | +27 pts |
| Citation precision | 0.49 | **0.78** | +29 pts |

## Quickstart

```bash
pip install -e ".[dev]"
python -m enterprise_rag.data        # generates 5,000 paragraphs + 1,000 Q&A
python -m enterprise_rag.models      # builds BM25 + TF-IDF dense + reranker artifacts
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python · pandas · scikit-learn · **rank_bm25** · TF-IDF (notebook fallback) · sentence-transformers + cross-encoder (production) · FastAPI · Next.js · Tailwind

## LLM-component note
The notebooks deliberately use a sklearn / TF-IDF / BM25 fallback so they run in Dataiku DSS without GPU or external API access. Production code in `src/enterprise_rag/` documents how to swap the answer generator for any hosted endpoint (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) — the retrieval and faithfulness scoring layers are model-agnostic.

## Author
Asad — MADS @ University of Michigan · Dubai HR
