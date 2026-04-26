"""Feature builders for the RAG retrieval + reranker stack.

Notebook-friendly: pure scikit-learn + rank_bm25. No external embedding API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class DenseEncoder:
    """TF-IDF + truncated SVD = a Dataiku-friendly LSA-style dense encoder."""

    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    norm: Normalizer

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self.tfidf.transform(texts)
        z = self.svd.transform(x)
        return self.norm.transform(z)


def build_dense_encoder(corpus_texts: list[str], n_components: int = 256) -> DenseEncoder:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z0-9]+\b",
    )
    x = tfidf.fit_transform(corpus_texts)
    n_components = min(n_components, min(x.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    svd.fit(x)
    norm = Normalizer(copy=False)
    norm.fit(svd.transform(x))
    return DenseEncoder(tfidf=tfidf, svd=svd, norm=norm)


def build_bm25(corpus_texts: list[str]) -> BM25Okapi:
    tokens = [tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokens)


def hybrid_score(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Min-max normalize each then weighted-sum."""
    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)
    return alpha * _norm(bm25_scores) + (1 - alpha) * _norm(dense_scores)


def candidate_features(
    query: str,
    candidates: pd.DataFrame,
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    median_len: float,
) -> pd.DataFrame:
    """Per-candidate features for the reranker LR."""
    q_lower = query.lower()
    feats = pd.DataFrame({
        "bm25_score": bm25_scores,
        "dense_score": dense_scores,
        "bm25_rank": np.argsort(np.argsort(-bm25_scores)),
        "dense_rank": np.argsort(np.argsort(-dense_scores)),
        "length_ratio": candidates["text"].str.split().apply(len) / max(median_len, 1.0),
        "domain_match": candidates["domain"].str.lower().apply(
            lambda d: 1 if d.split()[0] in q_lower else 0
        ).astype(int),
    })
    return feats


def expand_sources(source_str: str) -> list[tuple[str, int]]:
    out = []
    for tok in str(source_str).split(";"):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        d, p = tok.split(":")
        out.append((d, int(p)))
    return out
