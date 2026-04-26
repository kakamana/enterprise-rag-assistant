"""Train + persist the retrieval & rerank artifacts.

Produces under `models/`:
    bm25.pkl        — fitted BM25Okapi
    dense.pkl       — TF-IDF + SVD encoder + matrix
    reranker.pkl    — sklearn LogisticRegression
    corpus_index.parquet — paragraph table (mirrors data/processed/policy_corpus.parquet
                          but with row index aligned to the BM25 / dense matrices)

Run as:
    python -m enterprise_rag.models
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import (
    DenseEncoder,
    build_bm25,
    build_dense_encoder,
    candidate_features,
    expand_sources,
    hybrid_score,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_parquet(DATA / "policy_corpus.parquet").reset_index(drop=True)
    qa = pd.read_parquet(DATA / "policy_qa.parquet").reset_index(drop=True)
    return corpus, qa


def build_indexes(corpus: pd.DataFrame):
    texts = corpus["text"].tolist()
    bm25 = build_bm25(texts)
    enc = build_dense_encoder(texts)
    dense_mat = enc.transform(texts)
    return bm25, enc, dense_mat


def query_scores(
    query: str,
    bm25,
    enc: DenseEncoder,
    dense_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_toks = tokenize(query)
    bm25_s = np.asarray(bm25.get_scores(q_toks))
    q_dense = enc.transform([query])
    dense_s = (dense_mat @ q_dense.T).ravel()
    return bm25_s, dense_s


def gold_indices(corpus: pd.DataFrame, qa_row: pd.Series) -> list[int]:
    """Resolve `source_doc_ids` to row indices into the corpus DataFrame."""
    keyset = set(expand_sources(qa_row["source_doc_ids"]))
    if not keyset:
        return []
    mask = corpus.apply(
        lambda r: (r["doc_id"], int(r["paragraph_id"])) in keyset, axis=1
    )
    return list(np.where(mask.to_numpy())[0])


def make_training_pairs(
    qa: pd.DataFrame,
    corpus: pd.DataFrame,
    bm25,
    enc: DenseEncoder,
    dense_mat: np.ndarray,
    top_k: int = 20,
    neg_per_pos: int = 5,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    rows: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    median_len = float(np.median(corpus["text"].str.split().apply(len)))

    for _, qa_row in qa.iterrows():
        bm25_s, dense_s = query_scores(qa_row["question"], bm25, enc, dense_mat)
        hybrid = hybrid_score(bm25_s, dense_s, alpha=0.5)
        order = np.argsort(-hybrid)[:top_k]
        gold = set(gold_indices(corpus, qa_row))
        if not gold:
            continue

        pos_idx = [i for i in order if i in gold]
        neg_idx = [i for i in order if i not in gold]
        if not pos_idx:
            # Force-include gold even if it's not in top_k so the LR sees positives.
            pos_idx = [next(iter(gold))]
            neg_idx = [i for i in order if i != pos_idx[0]]
        if not neg_idx:
            continue
        rng.shuffle(neg_idx)
        neg_pick = neg_idx[: max(1, neg_per_pos * len(pos_idx))]
        pick = pos_idx + neg_pick
        candidates = corpus.iloc[pick].reset_index(drop=True)
        feats = candidate_features(
            qa_row["question"], candidates,
            bm25_s[pick], dense_s[pick], median_len,
        )
        y = np.array([1 if i in gold else 0 for i in pick], dtype=int)
        rows.append(feats)
        labels.append(y)

    X = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    y_arr = np.concatenate(labels) if labels else np.array([])
    return X, y_arr


def train_reranker(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
    clf.fit(X.values, y)
    return clf


def evaluate_recall_at_k(
    qa: pd.DataFrame,
    corpus: pd.DataFrame,
    bm25,
    enc: DenseEncoder,
    dense_mat: np.ndarray,
    reranker: LogisticRegression | None = None,
    alpha: float = 0.5,
    k: int = 5,
    top_k_candidates: int = 20,
) -> dict[str, float]:
    median_len = float(np.median(corpus["text"].str.split().apply(len)))
    hits = 0
    rr_sum = 0.0
    n = 0
    for _, qa_row in qa.iterrows():
        gold = set(gold_indices(corpus, qa_row))
        if not gold:
            continue
        bm25_s, dense_s = query_scores(qa_row["question"], bm25, enc, dense_mat)
        hybrid = hybrid_score(bm25_s, dense_s, alpha=alpha)
        order = np.argsort(-hybrid)[:top_k_candidates]
        if reranker is not None:
            cand = corpus.iloc[order].reset_index(drop=True)
            feats = candidate_features(
                qa_row["question"], cand, bm25_s[order], dense_s[order], median_len,
            )
            scores = reranker.predict_proba(feats.values)[:, 1]
            order = order[np.argsort(-scores)]
        topk = order[:k]
        if any(i in gold for i in topk):
            hits += 1
        # MRR
        rank = next(
            (j + 1 for j, i in enumerate(order[:10]) if i in gold), None
        )
        rr_sum += 1.0 / rank if rank else 0.0
        n += 1
    return {"recall@k": hits / max(n, 1), "mrr@10": rr_sum / max(n, 1), "n": float(n)}


def faithfulness(
    answer: str,
    sources: list[str],
    enc: DenseEncoder,
) -> float:
    if not sources:
        return 0.0
    a = enc.transform([answer])
    s = enc.transform(sources).mean(axis=0, keepdims=True)
    # encoder normalizes; s mean isn't unit so re-norm
    s = s / (np.linalg.norm(s) + 1e-9)
    return float((a @ s.T).ravel()[0])


def main() -> None:
    corpus, qa = load_data()
    print(f"corpus={len(corpus):,}  qa={len(qa):,}")

    bm25, enc, dense_mat = build_indexes(corpus)

    # 80/20 split for training the reranker
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(qa))
    cut = int(0.8 * len(qa))
    train_qa = qa.iloc[idx[:cut]].reset_index(drop=True)
    test_qa = qa.iloc[idx[cut:]].reset_index(drop=True)

    X, y = make_training_pairs(train_qa, corpus, bm25, enc, dense_mat)
    print(f"training pairs: X={X.shape} pos_rate={y.mean():.3f}")
    reranker = train_reranker(X, y)

    bm25_only = evaluate_recall_at_k(
        test_qa, corpus, bm25, enc, dense_mat, reranker=None, alpha=1.0
    )
    hybrid_only = evaluate_recall_at_k(
        test_qa, corpus, bm25, enc, dense_mat, reranker=None, alpha=0.5
    )
    full = evaluate_recall_at_k(
        test_qa, corpus, bm25, enc, dense_mat, reranker=reranker, alpha=0.5
    )
    print(f"BM25 only      Recall@5={bm25_only['recall@k']:.3f}  MRR@10={bm25_only['mrr@10']:.3f}")
    print(f"Hybrid (a=0.5) Recall@5={hybrid_only['recall@k']:.3f}  MRR@10={hybrid_only['mrr@10']:.3f}")
    print(f"Hybrid + rerank Recall@5={full['recall@k']:.3f}  MRR@10={full['mrr@10']:.3f}")

    joblib.dump(bm25, MODEL_DIR / "bm25.pkl")
    joblib.dump({"encoder": enc, "matrix": dense_mat}, MODEL_DIR / "dense.pkl")
    joblib.dump(reranker, MODEL_DIR / "reranker.pkl")
    corpus.to_parquet(MODEL_DIR / "corpus_index.parquet", index=False)
    print(f"saved artifacts -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
