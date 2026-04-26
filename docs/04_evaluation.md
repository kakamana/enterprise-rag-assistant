# Evaluation Plan — Enterprise RAG Assistant

## 1. Held-out test set
20% question-stratified split (200 of the 1,000 Q&A pairs), keeping the per-domain question count balanced.

## 2. Primary scorecard
| System | Recall@1 | Recall@5 | MRR@10 | Faithfulness | Citation precision |
|--------|---------|---------|--------|--------------|--------------------|
| BM25 only (baseline) | – | 0.61 | 0.42 | 0.55 | 0.49 |
| TF-IDF/LSA dense only | – | – | – | – | – |
| Hybrid (alpha-tuned) | – | – | – | – | – |
| Hybrid + sklearn LR rerank | – | **0.86** | **0.71** | **0.82** | **0.78** |

Numbers in the "Hybrid + rerank" row are the targets; populate with measured values after `python -m enterprise_rag.models` and notebook 04.

## 3. Slice analysis
- **By domain** — recall@5 reported per policy domain (HR, Finance, IT, ...).
- **By question length** — short (<= 8 tokens), medium (9-16), long (> 16).
- **By corpus density** — domains with more paragraphs vs. less.

## 4. Faithfulness calibration
- Reliability diagram over the held-out set: bin faithfulness scores in deciles, plot the empirical citation-precision per bin.
- Abstain threshold tuned so `abstain_rate in [0.05, 0.15]` while keeping faithfulness of the answered subset >= 0.8.

## 5. Robustness
- Drop a random 20% of paragraphs, re-index, re-evaluate (graceful degradation).
- Inject 10% question paraphrasing noise (synonym swap) — measure recall@5 decay.
- Inject 5% adversarial questions ("trick" questions whose answer is **not** in the corpus) — verify abstain rate jumps appropriately.

## 6. Error analysis
- Top-10 questions where Recall@5 = 0 — manual review for vocabulary mismatch vs. corpus gap.
- Faithfulness < 0.5 cases — ensure the system actually abstains, not answers.

## 7. Business impact simulation
Assume 5,000 internal policy queries / month, 30 sec average self-serve resolution vs. 8 minutes via ticket. Time saved ~ 625 hours / month at the target answered-rate.

## 8. Deployment readiness checklist
- [ ] Recall@5 >= 0.85 on held-out
- [ ] MRR@10 >= 0.70 on held-out
- [ ] Faithfulness mean >= 0.80 on answered subset
- [ ] Abstain rate calibrated to 5-15%
- [ ] Citation precision >= 0.75 on a 50-question manual review
- [ ] Per-domain recall gap < 0.10
- [ ] p95 latency < 300 ms on CPU
- [ ] Model card published at `mlops/model_card.md`
- [ ] UI: faithfulness gauge + sources panel visible on every answer
