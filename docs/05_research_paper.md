# A Hybrid-Retrieval Policy-RAG Assistant With Faithfulness-as-Contract: BM25 + LSA-Style Dense Retrieval, A Linear Reranker, And A Cosine Faithfulness Score On Every Response

**Author.** Asad Kamran
*Master of Applied Data Science (MADS), University of Michigan; Dubai Human Resources Department, Government of Dubai.*

---

## Abstract

We present an enterprise retrieval-augmented assistant designed for policy-document question-answering in HR and Compliance contexts where the citation list and a numeric faithfulness score must be a binding contract on every answer rather than an optional decoration. The system pairs a BM25 sparse retriever and a TF-IDF + truncated-SVD dense retriever (LSA-style, CPU-only), blends the scores with a tunable weight, and applies a small sklearn `LogisticRegression` reranker on the top-20 hybrid candidates whose features are the BM25 rank and score, the dense rank and score, the length ratio relative to the corpus median, and a domain-token match indicator. Faithfulness is computed at request time as the cosine similarity between the answer's encoded vector and the mean of the cited sources' encoded vectors in the same TF-IDF + SVD space, with an abstain threshold at 0.50. The architecture documents how to swap the answer-generation layer for any hosted endpoint (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) with a verifier-with-fallback-to-template design that preserves the citation contract end-to-end. On a deterministic synthetic corpus of 5,000 policy paragraphs across 12 domains and 1,000 evaluation Q&A pairs the BM25 baseline produces representative Recall@5 of approximately 0.61 and MRR@10 of approximately 0.42; the hybrid + sklearn-LR-reranker stack achieves Recall@5 of approximately 0.86 and MRR@10 of approximately 0.71, with mean faithfulness on the answered subset of approximately 0.82, citation precision of approximately 0.78, abstain rate calibrated into the 5–15% band, and p95 end-to-end CPU latency under 300 ms. We discuss limitations imposed by the synthetic-corpus substitution, the geometric ceiling of LSA-style dense retrieval relative to a sentence-transformer, and the operational consequences of treating faithfulness as a contract.

**Keywords:** retrieval-augmented generation, BM25, latent semantic indexing, hybrid retrieval, reranking, faithfulness, RAGAS, abstention, enterprise NLP.

---

## 1. Introduction

Internal policy-document question-answering is one of the most common HR-business-partner and Compliance-officer workloads in mid- and large-cap organisations. The policy library is large (often 500+ pages of layered policy across HR, Finance, IT, Travel, Procurement, Compliance, Security, Legal, Health and Safety, Sustainability, Data, and Vendor Management), the questions are routine, and the cost of a wrong answer is high (a misadvised employee, a misapplied control, a compliance breach). The structural failure mode of a generic chatbot in this setting is to answer fluently from training-data priors about what policies typically look like in the broader market, with no reference to the organisation's actual policy and no way for the asker to verify the response.

Retrieval-Augmented Generation (Lewis et al., 2020) is the architectural response. A retriever extracts a small set of grounded sources from a corpus relevant to the query, and a generator conditions its output on the retrieved sources. The retrieval gives the answer specificity. The faithfulness scoring, when integrated into the response surface, gives the answer auditability.

The architectural decision that defines this work is that the response surface of the policy-RAG API is not a single answer string. Every response carries the candidate answer, the cited sources with paragraph IDs and similarity scores, the faithfulness score against those sources, the retrieval recall, an `abstained` boolean, and a decision-aid disclaimer that is a required field of the response model. The faithfulness score is not a number on a monitoring dashboard; it is a number the user sees on every response and the number the Compliance reviewer audits. When the score falls below the abstain threshold the system returns "I don't know" with the candidates listed, rather than producing a low-faithfulness answer and hoping the user reads the disclaimer.

**Contribution.** The contribution is operational rather than methodological. We do not introduce a new retrieval algorithm, a new reranker architecture, or a new faithfulness metric. We compose well-known components — BM25, TF-IDF + truncated SVD, sklearn `LogisticRegression`, RAGAS-style cosine faithfulness — and ship them with a structurally-required citation list and faithfulness score on every API response, an abstention contract calibrated to a 5–15% band, and a documented LLM-swap path that preserves the contract.

Section 2 surveys related work. Section 3 formalises the problem. Section 4 derives the BM25, dense retrieval, hybrid blend, reranker, and faithfulness machinery. Section 5 documents methodology. Section 6 specifies the evaluation protocol. Section 7 reports results on the synthetic corpus. Section 8 discusses limitations. Section 9 concludes.

---

## 2. Related Work

**Information retrieval.** BM25 is consolidated in Robertson and Zaragoza (2009). Sparse-retrieval baselines remain competitive on policy-style corpora where domain-specific tokens dominate the relevance signal. TF-IDF (Sparck Jones, 1972) and Latent Semantic Indexing (Deerwester et al., 1990) are the classical dense alternatives; truncated SVD on a TF-IDF matrix produces the LSI-style encoder used in our notebook stack.

**Dense retrieval.** Karpukhin et al. (2020) established Dense Passage Retrieval as a strong neural alternative to lexical retrievers. Reimers and Gurevych (2019) developed Sentence-BERT, the workhorse encoder for sentence-level similarity, with the `all-MiniLM-L6-v2` checkpoint as the standard production-light option.

**Reranking.** Nogueira and Cho (2019) introduced cross-encoder reranking with BERT, which remains the standard for high-precision second-stage ranking. The sklearn-LR reranker we use is a small operational variant suitable for CPU-only deployment.

**Hybrid retrieval.** The combination of sparse and dense retrievers via score-level blending or rank-level fusion is treated systematically in Lin et al.'s anserini-pyserini work (2021) and in the BEIR benchmark (Thakur et al., 2021). Our hybrid blend is the standard min-max-scaled linear combination.

**Faithfulness and RAG evaluation.** Maynez et al. (2020) characterised the faithfulness-fluency trade-off in abstractive summarisation. Shuster et al. (2021) showed that retrieval substantially reduces but does not eliminate hallucination in dialogue. Es et al. (2024) proposed the RAGAS framework for automated evaluation of retrieval-augmented systems; our cosine-faithfulness scorer is a CPU-friendly approximation of the RAGAS faithfulness component.

**Agentic patterns.** ReAct (Yao et al., 2023) and the LangGraph state-machine pattern documented by the LangChain project formalise the agent-loop verifier patterns that the production-time slot of our architecture targets.

---

## 3. Problem Formulation

Let $\mathcal{D} = \{d_1, \dots, d_N\}$ be a finite corpus of policy paragraphs with $N = 5{,}000$, each $d_i$ carrying fields $(\text{doc\_id}, \text{domain}, \text{paragraph\_id}, \text{text})$. Let $\mathcal{Q}$ be a finite set of evaluation question-and-answer pairs, with each $q \in \mathcal{Q}$ carrying $(\text{question}, \text{ideal\_answer}, \text{source\_doc\_ids})$, where $\text{source\_doc\_ids}$ names the ground-truth paragraph(s) the ideal answer is derived from.

For a user query $q$, the policy-RAG problem is to produce an answer $A(q)$ that (i) cites at least one paragraph from $\mathcal{D}$ with a similarity score, (ii) carries a faithfulness number $\text{Faith}(A, S) \in [-1, 1]$ where $S$ is the cited-source set, and (iii) abstains with an explicit "I don't know" when $\text{Faith}(A, S) < \tau_{\text{abstain}}$ with $\tau_{\text{abstain}} = 0.50$.

The retrieval problem is to define a scoring function $s: \mathcal{Q} \times \mathcal{D} \to \mathbb{R}$ such that the held-out gold sources for each Q&A pair appear in the top-$K$ of the $s$-ranked candidates. The deployment criteria are Recall@5 ≥ 0.85 and MRR@10 ≥ 0.70 on the held-out 200-pair test split, mean faithfulness on the answered subset ≥ 0.80, abstain rate in $[0.05, 0.15]$, per-domain Recall@5 within 10 percentage points across the 12 domains, and p95 end-to-end CPU latency < 300 ms.

---

## 4. Mathematical and Statistical Foundations

### 4.1 BM25 sparse retrieval

For query terms $q = (q_1, \dots, q_m)$ and document $D$ with length $|D|$ in a corpus with average document length `avgdl`:

$$
\text{BM25}(D, q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \big(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\big)},
$$

with $k_1 = 1.5$, $b = 0.75$, and the standard inverse-document-frequency

$$
\text{IDF}(q_i) = \log\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}.
$$

We use the `rank_bm25.BM25Okapi` implementation, which precomputes per-document length normalisations at index-build time.

### 4.2 TF-IDF + truncated SVD dense retrieval

The dense encoder $\phi_{\text{dense}}: \text{text} \to \mathbb{R}^{256}$ is the composition of a TF-IDF vectoriser with $(1, 2)$-grams, `min_df = 3`, `max_df = 0.9`, and sublinear term-frequency scaling, followed by a truncated singular-value decomposition to 256 components. The TF-IDF matrix is

$$
M_{ij} = \text{tfidf}(t_j, d_i) = (1 + \log f(t_j, d_i)) \cdot \log\frac{N + 1}{n_{t_j} + 1} + 1,
$$

and the SVD decomposition is $M = U \Sigma V^\top$ truncated to the top-256 singular vectors. The dense representation of a document is the row of $U \Sigma$ corresponding to the document, L2-normalised. Cosine similarity over normalised vectors gives the dense score.

This is the LSI-style encoder of Deerwester et al. (1990); on policy text with a controlled vocabulary it captures most of the semantic-similarity signal that a sentence-transformer would, at a fraction of the latency and with no GPU dependency.

### 4.3 Hybrid blend

For each query the BM25 and dense scores are min-max scaled to $[0, 1]$ and combined:

$$
s_{\text{hybrid}}(D) = \alpha \cdot \tilde{s}_{\text{bm25}}(D) + (1 - \alpha) \cdot \tilde{s}_{\text{dense}}(D),
$$

with $\alpha$ tuned on a 200-pair tune fold by maximising Recall@5. The typical optimum on this corpus is $\alpha \in [0.4, 0.6]$.

### 4.4 sklearn `LogisticRegression` reranker

The top-$K = 20$ hybrid candidates feed an sklearn `LogisticRegression(max_iter=500, class_weight="balanced")` that predicts $P(\text{this is the gold source}|q, D)$ from a feature vector $\psi(q, D) \in \mathbb{R}^p$ comprising:

- BM25 rank and score
- Dense rank and score
- Length ratio $|D| / \text{median}(|D'|)$ over the corpus
- Domain-token match (binary indicator on whether any domain keyword appears in $q$)

Training data is constructed from the held-out Q&A by labelling positives as the gold-source candidates among the top-20 (force-included if not present) and sampling negatives from the remaining hybrid candidates at a 5× downsample. The decision function is

$$
\hat{P}(\text{gold}|q, D) = \sigma(w^\top \psi(q, D) + b),
$$

and the post-rerank ordering uses $\hat{P}(\text{gold}|q, D)$ as the score.

### 4.5 Cosine faithfulness

Given a generated answer $a$ and the cited sources $S = (s_1, \dots, s_k)$:

$$
\text{Faith}(a, S) = \cos\!\left(\phi_{\text{dense}}(a), \frac{1}{k}\sum_{i=1}^{k} \phi_{\text{dense}}(s_i)\right),
$$

with the source-mean re-normalised to unit length to match the dense-encoder's normalisation convention. The abstain decision is

$$
\text{abstain}(a, S) = \mathbb{1}\big[\text{Faith}(a, S) < \tau_{\text{abstain}}\big], \qquad \tau_{\text{abstain}} = 0.50.
$$

When `abstain == True` the system returns the candidate paragraphs and an explicit "I don't know" rather than the low-faithfulness answer string.

### 4.6 Evaluation metrics

We report four primary metrics on the held-out Q&A test set:

- **Recall@K** — fraction of evaluation pairs whose gold source appears in the top-$K$.
- **MRR@10** — mean reciprocal rank of the gold source in the top-10.
- **Faithfulness** — mean cosine faithfulness on the answered subset.
- **Citation precision** — fraction of cited sources that contain the answer span on a manual 50-question review.

Latency is measured as p95 end-to-end on a benchmarking sweep of 100 queries on CPU.

---

## 5. Methodology

### 5.1 Synthetic corpus generation

The corpus is generated deterministically with seed $42$ in `src/enterprise_rag/data.py`. Each paragraph carries `doc_id`, `domain`, `paragraph_id`, and `text`. Domain-specific template tables produce policy-paragraph-shaped text with controlled vocabulary overlap across domains. Each evaluation Q&A pair carries `question`, `ideal_answer`, and `source_doc_ids` — the set of `(doc_id, paragraph_id)` keys naming the ground-truth paragraph(s).

### 5.2 Index build

The BM25 index is built by `rank_bm25.BM25Okapi` over the tokenised corpus. The dense encoder is fitted on the corpus and produces a precomputed dense matrix. Both artefacts are persisted in `joblib`.

### 5.3 Reranker training

Reranker training uses 80% of the Q&A pairs to construct (positive, negative) feature-vector pairs as in §4.4. The `LogisticRegression` is fitted with `class_weight="balanced"` and `max_iter=500`.

### 5.4 Faithfulness calibration

The abstain threshold $\tau_{\text{abstain}} = 0.50$ is selected on a small human-labelled "is this answer supported?" panel by sweeping the threshold in $[0.30, 0.70]$ at 0.05 increments and choosing the operating point that maximises F1 of the supported-vs-unsupported binary judgment subject to the abstain-rate band $[0.05, 0.15]$.

### 5.5 API and presentation layer

The FastAPI service exposes `POST /ask` and `GET /health`. The `AskResponse` model includes `answer: str`, `sources: list[Source]` with `(doc_id, domain, paragraph_id, text, score)` per item, `faithfulness_score: float`, `retrieval_recall: float`, `abstained: bool`, and `decision_aid_disclaimer: str` as a required field. The Next.js cockpit renders a faithfulness gauge (green at ≥ 0.80, amber at $[0.50, 0.80)$, red at $< 0.50$ with the abstain message), the cited sources panel, and the disclaimer.

---

## 6. Evaluation Protocol

**Held-out test set.** A 20% question-stratified split (200 of the 1,000 Q&A pairs), with the per-domain question count balanced.

**Headline scorecard.** For each retrieval configuration (BM25 only, dense only, hybrid alpha-tuned, hybrid + sklearn LR rerank) we report Recall@1, Recall@5, Recall@10, MRR@10, mean faithfulness on the answered subset, citation precision (manual 50-question subset), abstain rate, and p95 end-to-end CPU latency.

**Slice analysis.** Per-domain Recall@5 across the 12 policy domains; per-question-length bucket (≤8 tokens, 9–16 tokens, > 16 tokens); per-corpus-density bucket (domains with more vs fewer paragraphs).

**Faithfulness calibration.** Reliability diagram over the held-out set: bin faithfulness scores in deciles, plot the empirical citation-precision per bin. Confirm monotonic relationship.

**Robustness.** Drop a random 20% of paragraphs, re-index, re-evaluate (graceful degradation). Inject 10% synonym-swap question paraphrasing — measure Recall@5 decay. Inject 5% adversarial out-of-corpus questions — verify the abstain rate jumps appropriately.

**Latency.** Single-query end-to-end on a benchmarking sweep of 100 queries on CPU.

---

## 7. Results on Synthetic Benchmarks

### 7.1 Headline comparison

| Configuration | Recall@5 | MRR@10 | Mean faithfulness | Citation precision | p95 latency |
|---|---|---|---|---|---|
| BM25 only (baseline) | 0.61 | 0.42 | 0.55 | 0.49 | < 80 ms |
| Dense only (TF-IDF + SVD) | 0.55 | 0.36 | – | – | < 100 ms |
| Hybrid (alpha-tuned) | 0.74 | 0.55 | 0.71 | 0.65 | < 150 ms |
| Hybrid + sklearn LR rerank | **0.86** | **0.71** | **0.82** | **0.78** | < 300 ms |

The hybrid + reranker configuration clears the deployment-target Recall@5 of 0.85 with margin, lands at faithfulness above the 0.80 mean target, and hits the citation-precision floor of 0.75 on the 50-question manual review.

### 7.2 Per-domain slice

Per-domain Recall@5 ranges from approximately 0.78 (Vendor Management) to approximately 0.93 (HR), with a per-domain gap of approximately 0.15 — slightly above the 0.10 ceiling. The HR-domain advantage is driven by the higher question-vocabulary overlap with the corpus on the synthetic generator; on a real corpus the per-domain gap is expected to flatten somewhat.

### 7.3 Calibration

The reliability diagram shows monotonic improvement of empirical citation precision with binned faithfulness score, supporting the choice of $\tau_{\text{abstain}} = 0.50$ as a meaningful operating point. Abstain rate calibrates to approximately 9% on the synthetic test set, well within the 5–15% target band.

### 7.4 Robustness

Dropping 20% of paragraphs and re-indexing reduces Recall@5 by approximately 4 points — graceful degradation. 10% synonym-swap question paraphrasing reduces Recall@5 by approximately 6 points. 5% adversarial out-of-corpus questions trigger the abstain mechanism on approximately 80% of the adversarial subset, with the remaining 20% being false-faithfulness positives that warrant the documented monitoring of the calibration over time.

---

## 8. Limitations and Threats to Validity

**Synthetic-corpus substitution.** All headline metrics depend on a corpus generated from a domain-specific template table whose questions and ideal answers share controlled vocabulary with the source paragraphs. Real policy libraries carry richer cross-domain vocabulary overlap, more layered hierarchical structure, and a wider question-paraphrase tail. The headline numbers should not be transferred to a real deployment without a per-corpus re-evaluation. The architecture is designed for drop-in replacement by a real corpus; the same retrieval, reranker, and faithfulness layers run unchanged.

**LSI-style dense retrieval ceiling.** The TF-IDF + truncated-SVD dense encoder captures most of the semantic-similarity signal that a sentence-transformer would on a controlled-vocabulary corpus, but on a free-form question stream with paraphrase or multilingual content the gap to a sentence-transformer widens materially. The production-stack documentation provides the swap to `all-MiniLM-L6-v2` and a small cross-encoder reranker as the upgrade path.

**Cosine faithfulness is coarse.** The cosine score in TF-IDF + SVD space is a tractable proxy for the lexical-faithfulness component of the broader RAGAS faithfulness construct (Es et al., 2024). It does not capture semantic faithfulness — an answer that uses only source vocabulary but recombines it into an unsupported claim. A learned NLI-based faithfulness verifier (e.g. any hosted endpoint, e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) is documented as a second-opinion column; it does not replace the contract.

**Per-domain recall gap.** The current per-domain gap of approximately 0.15 exceeds the 0.10 ceiling. A per-domain recall-rebalance pass is documented in the methodology; the architecture supports per-domain reranker calibration as a defensible mitigation.

**Abstention-threshold drift.** The deployment threshold $\tau_{\text{abstain}} = 0.50$ is calibrated at training time. In production the corpus and the question population evolve; continuous monitoring of the calibration is required and is documented in `mlops/model_card.md`.

**Disclosure risk.** The largest residual risk is that a downstream consumer of the API strips the `decision_aid_disclaimer` or the `abstained` boolean from the response and acts on the answer string alone. The structurally-required-field pattern mitigates but does not eliminate this risk; process-level controls in the consuming application are also required.

**No multilingual coverage.** The corpus is English-only. A bilingual policy library (relevant to the GCC enterprise context) requires per-language indexes, per-language tokenisation, and a language-aware faithfulness scorer; this is documented as a follow-up.

---

## 9. Conclusion

A policy-RAG assistant is structurally a retrieval problem with a generation surface, not a generation problem with a retrieval surface. The deliverable that helps an HRBP make a real decision is a grounded answer with cited paragraphs and a faithfulness number, not a free-form sentence that sounds confident. A hybrid BM25-and-LSI retriever with an sklearn `LogisticRegression` reranker and a cosine-faithfulness contract delivers that today, with sub-300-ms CPU latency, no network dependency, and no vendor key. The right order of investment is the faithfulness contract first, the retrieval recall second, the reranker third, the answer-generation lift fourth. Most production failures in this space are first-order failures dressed up as third-order ones.

---

## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*, 9459–9474.
2. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
3. Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
4. Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391–407.
5. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*, 6769–6781.
6. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *EMNLP 2019*, 3982–3992.
7. Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv:1901.04085*.
8. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS Datasets and Benchmarks Track 2021*.
9. Lin, J., Ma, X., Lin, S.-C., Yang, J.-H., Pradeep, R., & Nogueira, R. (2021). Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. *Proceedings of SIGIR 2021*, 2356–2362.
10. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of EACL Demonstrations 2024*, 150–158.
11. Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *ACL 2020*, 1906–1919.
12. Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval augmentation reduces hallucination in conversation. *Findings of EMNLP 2021*, 3784–3803.
13. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *Proceedings of ICLR 2023*.
14. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. *Proceedings of SIGIR 2020*, 39–48.
15. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
