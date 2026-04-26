"""Inference wrapper for the RAG API.

Loads BM25 + dense + reranker artifacts and exposes `answer(question)` returning
{answer, sources, faithfulness_score, retrieval_recall_at_5}.

The notebook stack constructs the answer as a *templated extractive snippet* —
the top reranked paragraph trimmed to its first 2 sentences. Production callers
can swap this for any hosted endpoint (e.g. GPT-4o-mini, Mistral-Large,
Llama-3-Instruct) by replacing `_compose_answer`. Retrieval and faithfulness
scoring are model-agnostic.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import models
from .features import candidate_features

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

ABSTAIN_THRESHOLD = 0.50
TOP_K_CANDIDATES = 20
TOP_K_RETURN = 5


@lru_cache(maxsize=1)
def _load_artifacts():
    bm25 = joblib.load(MODEL_DIR / "bm25.pkl")
    dense = joblib.load(MODEL_DIR / "dense.pkl")
    reranker = joblib.load(MODEL_DIR / "reranker.pkl")
    corpus = pd.read_parquet(MODEL_DIR / "corpus_index.parquet")
    median_len = float(np.median(corpus["text"].str.split().apply(len)))
    return dict(
        bm25=bm25,
        encoder=dense["encoder"],
        dense_mat=dense["matrix"],
        reranker=reranker,
        corpus=corpus,
        median_len=median_len,
    )


def _compose_answer(top_text: str) -> str:
    """Templated extractive answer: first 2 sentences of the top reranked paragraph.

    Production override point: replace this with any hosted endpoint
    (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) prompted with the
    retrieved passages.
    """
    sentences = re.split(r"(?<=[.!?])\s+", top_text.strip())
    return " ".join(sentences[:2]).strip()


def answer(question: str, k: int = TOP_K_RETURN) -> dict:
    art = _load_artifacts()
    bm25 = art["bm25"]
    enc = art["encoder"]
    dense_mat = art["dense_mat"]
    reranker = art["reranker"]
    corpus = art["corpus"]

    bm25_s, dense_s = models.query_scores(question, bm25, enc, dense_mat)
    hybrid = models.hybrid_score(bm25_s, dense_s, alpha=0.5)
    order = np.argsort(-hybrid)[:TOP_K_CANDIDATES]
    cand = corpus.iloc[order].reset_index(drop=True)
    feats = candidate_features(
        question, cand, bm25_s[order], dense_s[order], art["median_len"],
    )
    rerank_scores = reranker.predict_proba(feats.values)[:, 1]
    rerank_order = order[np.argsort(-rerank_scores)]
    top_idx = rerank_order[:k]

    sources = []
    for idx in top_idx:
        row = corpus.iloc[idx]
        sources.append({
            "doc_id": row["doc_id"],
            "domain": row["domain"],
            "paragraph_id": int(row["paragraph_id"]),
            "text": row["text"],
            "score": float(hybrid[idx]),
        })

    composed = _compose_answer(corpus.iloc[top_idx[0]]["text"])
    faith = models.faithfulness(composed, [s["text"] for s in sources], enc)

    if faith < ABSTAIN_THRESHOLD:
        composed = (
            "I do not have enough confidence in the supporting documents to answer."
            " Please escalate to the policy owner."
        )

    # retrieval_recall is the fraction of returned sources whose hybrid score is
    # at least half of the top score — a proxy of "did we retrieve a tight cluster".
    top_score = sources[0]["score"] if sources else 0.0
    if top_score > 0:
        tight = sum(1 for s in sources if s["score"] >= 0.5 * top_score) / len(sources)
    else:
        tight = 0.0

    return {
        "answer": composed,
        "sources": sources,
        "faithfulness_score": float(faith),
        "retrieval_recall": float(tight),
        "abstained": faith < ABSTAIN_THRESHOLD,
    }
