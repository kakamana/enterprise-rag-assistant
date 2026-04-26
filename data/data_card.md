# Data Card — #19 Enterprise RAG Assistant

## Dataset composition

| Layer | Source | Rows x cols | Purpose |
|-------|--------|-------------|---------|
| Corpus | Synthetic (12 policy domains) | 5,000 x 4 | Retrievable policy paragraphs |
| Q&A | Synthetic (1 question per ~5 paragraphs) | 1,000 x 4 | Held-out evaluation with ground-truth source citations |

## Schema

### `policy_corpus.parquet`
| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Stable doc ID, e.g. `HR-007` |
| `domain` | str | One of 12 domains |
| `paragraph_id` | int | Sequential paragraph index inside the doc |
| `text` | str | Paragraph text (~30-90 words) |

### `policy_qa.parquet`
| Column | Type | Description |
|--------|------|-------------|
| `qa_id` | str | Stable QA ID |
| `question` | str | Natural-language question |
| `answer` | str | Reference answer text |
| `source_doc_ids` | str | `;`-separated list of `doc_id:paragraph_id` ground-truth sources |

## Policy domains

`HR, Finance, IT, Travel, Procurement, Compliance, Security, Legal, Health & Safety, Sustainability, Data, Vendor`

## Known biases
- Synthetic corpus uses formal English register only; multi-language and dialectal variation are out of scope.
- The Q&A generator pairs each question with 1-2 ground-truth paragraphs; multi-hop questions (>2 sources) are absent.
- Length distribution is bounded by the templating; real policies will have longer outliers.

## PII
None. Names, emails, employee IDs, vendor names — all placeholder strings.

## Splits
- Train (for the reranker LR) 60% · val 20% · test 20%
- Stratified on `domain`

## Reproducing
```bash
python -m enterprise_rag.data
```
Deterministic seed = 42 (`enterprise_rag.data.SEED`).

## Licensing
- Synthetic corpus + Q&A — MIT (this repo).
