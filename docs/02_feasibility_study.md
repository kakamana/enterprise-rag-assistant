# Feasibility Study — Enterprise RAG Assistant

## 1. Data feasibility

### Primary: synthetic 12-domain policy corpus
- **Generator:** `src/enterprise_rag/data.py::build_corpus()` produces 5,000 paragraphs across HR, Finance, IT, Travel, Procurement, Compliance, Security, Legal, Health & Safety, Sustainability, Data, and Vendor Management.
- **Schema:** `doc_id, domain, paragraph_id, text`.
- **Q&A:** `build_qa()` produces 1,000 question / answer / `source_doc_ids` triples that reference the corpus by ID, giving every evaluation question a verifiable ground truth.
- **Why synthetic:** real enterprise policies are sensitive; the synthetic corpus uses structural templates that mimic policy register (numbered clauses, "Employees must...", "Reimbursement is capped at...") so retrieval models behave realistically.

### Augmentation hooks
- Drop a real PDF dump under `data/raw/policies/` and `enterprise_rag.data` will skip the synthetic generator (documented in `data/data_sources.md`).

## 2. Technical feasibility
- **Retrieval shortlist**
  - Sparse: BM25 via `rank_bm25` (default Okapi BM25, k1=1.5, b=0.75).
  - Dense (notebook fallback): TF-IDF + truncated SVD (LSA-style) — Dataiku-friendly.
  - Dense (production): sentence-transformers `bge-small-en` or `e5-small-v2`.
- **Reranker**
  - Notebook fallback: scikit-learn logistic regression on `[bm25_score, tfidf_score, length_ratio, domain_match]` features.
  - Production: cross-encoder `ms-marco-MiniLM-L-6-v2`.
- **Faithfulness**
  - Cosine similarity between the (TF-IDF / dense) embedding of the answer and the union of retrieved sources.
- **Compute:** 1 CPU; index build < 30 s; query latency < 50 ms (notebook stack).
- **Serving:** FastAPI; artifacts ~10 MB.

## 3. Economic feasibility
| Line item | Monthly cost |
|-----------|--------------|
| 1x small container | ~$8 |
| Storage | ~$1 |
| Optional hosted-LLM tokens | metered, ~$5-50 |
| **Total** | **~$14-60 / mo** |

**Value:** even a 30-second saving per query × ~5,000 employee queries / month is hundreds of hours / month redirected away from policy hunting.

## 4. Operational feasibility
- **Indexing:** rebuild on every policy publication; CI job runs the data + models scripts.
- **Monitoring:** weekly recall@5 sample; per-domain breakdown; drift alarm on faithfulness mean.
- **Human-in-the-loop:** abstain UX surfaces "I don't know — escalate to HR" instead of guessing.

## 5. Ethical / legal feasibility
- **Right to explanation:** every answer cites paragraph IDs + ships a faithfulness score the user can act on.
- **No personal data** in the corpus; the faithfulness metric never logs the question alongside identifiers.
- **Vendor-neutral generation:** the prod API can be wired to any hosted endpoint (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) or a local model — no lock-in.

## 6. Recommendation
**Go.** The retrieval and faithfulness layers are sufficient on their own to deliver value (you already see the right paragraph). Generation is a configurable upgrade, not a dependency.
