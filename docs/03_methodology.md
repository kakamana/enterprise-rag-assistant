# Methodology — Enterprise RAG Assistant

Three layers, one stack:
1. **Retrieve** — BM25 (sparse) + TF-IDF/LSA (dense fallback) + sentence-encoder (production), blended.
2. **Rerank** — sklearn LR on retrieval features (notebook) / cross-encoder (production).
3. **Score faithfulness** — cosine similarity between the answer and the union of cited sources.

The notebook stack is the Dataiku-compatible fallback: BM25 + TF-IDF + sklearn LR. Production code in `src/enterprise_rag/` documents how to swap in any hosted endpoint (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) for the answer-generation step. **Retrieval and faithfulness are model-agnostic.**

---

## 1. EDA plan
- Paragraph length distribution by domain.
- Question length distribution.
- Vocabulary overlap between questions and source paragraphs (a sanity proxy for retrieval difficulty).
- Domain balance — every domain should contribute ~ 1/12 of the corpus.

## 2. Retrieval — sparse (BM25)
For query terms $q = (q_1, \ldots, q_m)$ and document $D$ with length $|D|$:

$$ \text{BM25}(D, q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $$

with $k_1 = 1.5$, $b = 0.75$, $\text{IDF}(q_i) = \log\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$.

## 3. Retrieval — dense (notebook fallback)
TF-IDF (word n-grams 1-2) + truncated SVD to 256 dims = an LSA-style dense retriever that runs on CPU and ships in `joblib`. Cosine similarity over normalized vectors.

## 4. Hybrid blend
For each query we min-max scale BM25 and dense scores to $[0, 1]$ and combine:

$$ s_{\text{hybrid}}(D) = \alpha \cdot s_{\text{bm25}}(D) + (1 - \alpha) \cdot s_{\text{dense}}(D) $$

$\alpha$ is tuned on the held-out Q&A by maximizing Recall@5 (typical optimum ~ 0.4-0.6).

## 5. Reranker (notebook)
Top-K=20 hybrid candidates feed an sklearn logistic regression that predicts P(this is the gold source) from features:
- BM25 rank and score
- Dense rank and score
- Length ratio (paragraph / median)
- Domain-token match (1 if any domain keyword appears in the query)

Trained on the held-out Q&A with positive = gold source, negatives = the other 19 candidates (5x downsample).

## 6. Faithfulness (RAGAS-style)
Given a generated answer $a$ and the cited sources $S = (s_1, \ldots, s_k)$:

$$ \text{Faith}(a, S) = \cos\!\left(\phi(a), \sum_{i} \phi(s_i) / k\right) $$

where $\phi(\cdot)$ is the same TF-IDF + SVD encoder used for dense retrieval. Calibrated against a small human-labeled "is this answer supported?" set; threshold for **abstain** = 0.50.

## 7. Cross-validation
- **Question-bucketed 5-fold** on the 1,000 Q&A pairs, stratified by domain.
- The reranker is retrained on each training fold; alpha is selected on a 200-pair tune fold.

## 8. Evaluation metrics
| Layer | Primary | Secondary |
|-------|---------|-----------|
| Retrieval | Recall@5, MRR@10 | Recall@1, Recall@10 |
| Reranker | nDCG@5 | hit@1 lift over hybrid-only |
| End-to-end | Faithfulness, Citation precision | Abstain rate, latency p95 |

## 9. Interpretability
- Per-query: show the retrieved-paragraph IDs and their hybrid scores.
- Per-domain: recall slice in the eval notebook.
- UI faithfulness gauge (green >= 0.8, amber 0.5-0.8, red < 0.5).

## 10. References
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA*, 2020.
- Nogueira & Cho, *Passage Re-ranking with BERT*, 2019.
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, 2023.
- Reimers & Gurevych, *Sentence-BERT*, 2019.
