# Business Requirements — Enterprise RAG Assistant

## 1. Problem Statement
Employees and Compliance staff routinely waste hours hunting for the right paragraph inside hundreds of policy PDFs spread across HR, Finance, IT, Travel, Procurement, Compliance, Security, Legal, Health & Safety, Sustainability, Data, and Vendor Management. Generic chat tools answer fluently but cite nothing, so the user has no way to verify the response. A grounded RAG assistant that returns **answer + cited paragraphs + a faithfulness score** turns the same query into a defensible, audit-ready answer in under a second.

## 2. Stakeholders
| Role | Interest | Success criterion |
|------|----------|-------------------|
| Employee | Quick policy answer | Self-serve resolution; no ticket required |
| HR Business Partner | Consistent guidance | Answers cite the same authoritative paragraph |
| Compliance Officer | Auditable trail | Every answer carries source paragraph IDs + faithfulness score |
| IT / Security | No data leakage | All retrieval and (optional) generation runnable on-prem |
| Internal Audit | Reproducibility | Identical question returns identical citations |

## 3. Business Objectives
1. Achieve **Recall@5 >= 0.85** on the held-out Q&A set (i.e. one of the top-5 retrieved paragraphs is the ground-truth source 85% of the time).
2. Achieve **faithfulness >= 0.80** on average (cosine between answer and best-supporting source).
3. Return **citations on 100%** of answered questions; abstain ("I don't know") when faithfulness < 0.5.
4. **p95 latency < 300 ms** end-to-end (retrieval + rerank + faithfulness scoring).

## 4. KPIs
| KPI | Definition | Target | Baseline |
|-----|-----------|--------|----------|
| Recall@5 | Ground-truth source in top-5 retrieved | >= 0.85 | 0.61 (BM25) |
| MRR@10 | Mean reciprocal rank of GT source | >= 0.70 | 0.42 |
| Faithfulness | Cosine(answer, retrieved sources) | >= 0.80 | 0.55 |
| Citation precision | Cited sources actually contain answer span | >= 0.75 | 0.49 |
| Abstain rate | Calibrated "I don't know" rate | 5-15% | n/a |

## 5. Scope
**In scope:** the synthetic 12-domain corpus (5,000 paragraphs) and held-out Q&A (1,000 pairs); hybrid retrieval; cross-encoder reranking; faithfulness scoring; FastAPI + Next.js cockpit.
**Out of scope:** cross-language retrieval, OCR ingestion, conversational multi-turn memory, automated policy authoring.

## 6. Constraints & Assumptions
- **Data sovereignty:** retrieval runs entirely on-prem; the optional answer generator is configurable to any hosted endpoint or local model.
- **Auditability:** every prediction must persist `{question, retrieved_ids, answer, faithfulness}` to an audit log.
- **Latency:** p95 < 300 ms on CPU-only.
- **Reproducibility:** fixed-seed corpus + Q&A generator (`python -m enterprise_rag.data`).

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hallucination in the generated answer | High | High | Faithfulness gate + abstain when score < 0.5 |
| Stale policy paragraphs | Medium | High | Index versioning; retrieval result includes paragraph version |
| Domain drift across the 12 policy areas | Medium | Medium | Per-domain recall slice in CI |
| Over-trust by users | Medium | Medium | UI shows source paragraph excerpts + faithfulness gauge |

## 8. Timeline
- **Week 1** — Synthetic corpus + Q&A generator; EDA on length / domain balance
- **Week 2** — BM25 + TF-IDF dense; hybrid blend tuning
- **Week 3** — Cross-encoder reranker + faithfulness scoring
- **Week 4** — FastAPI + Next.js chat cockpit; abstain logic
- **Week 5** — Content (Medium / LinkedIn / YouTube); ship
