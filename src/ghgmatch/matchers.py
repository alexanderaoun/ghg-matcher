from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_union

from .catalog import CATALOG, catalog_documents
from .text import normalize, with_vendor

CODES = [c.code for c in CATALOG]


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max(axis=1, keepdims=True)) / temperature
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Matcher(ABC):
    name: str = "matcher"
    needs_training: bool = False

    def fit(self, texts: list[str], vendors: list[str], labels: list[str]) -> Matcher:
        return self

    @abstractmethod
    def scores(self, texts: list[str], vendors: list[str]) -> np.ndarray: ...

    def predict_proba(self, texts: list[str], vendors: list[str]) -> np.ndarray:
        return self.scores(texts, vendors)


class FuzzyCatalogMatcher(Matcher):
    name = "fuzzy · catalog"

    def __init__(self, temperature: float = 8.0):
        self.temperature = temperature
        self.docs = [normalize(d) for d in catalog_documents()]

    def scores(self, texts, vendors):
        queries = [normalize(t) for t in texts]
        raw = process.cdist(queries, self.docs, scorer=fuzz.token_set_ratio, workers=-1)
        return _softmax(np.asarray(raw, dtype=float), self.temperature)


class TfidfCatalogMatcher(Matcher):
    def __init__(self, analyzer: str = "char_wb", ngram_range=(3, 5), temperature: float = 0.05):
        self.name = f"tf-idf {'char' if analyzer != 'word' else 'word'} · catalog"
        self.temperature = temperature
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True)
        docs = [normalize(d) for d in catalog_documents()]
        self.doc_matrix = self.vectorizer.fit_transform(docs)

    def scores(self, texts, vendors):
        q = self.vectorizer.transform([normalize(t) for t in texts])
        return _softmax(cosine_similarity(q, self.doc_matrix), self.temperature)


class EmbeddingCatalogMatcher(Matcher):
    name = "embeddings · catalog"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", temperature: float = 0.05):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.temperature = temperature
        self.doc_emb = self.model.encode(catalog_documents(), normalize_embeddings=True)

    def scores(self, texts, vendors):
        q = self.model.encode(list(texts), normalize_embeddings=True, batch_size=64)
        return _softmax(q @ self.doc_emb.T, self.temperature)


class VendorPriorMatcher(Matcher):
    name = "vendor lookup · history"
    needs_training = True

    def fit(self, texts, vendors, labels):
        counts: dict[str, Counter] = defaultdict(Counter)
        for v, y in zip(vendors, labels):
            counts[normalize(v)][y] += 1
        self.table = {v: c for v, c in counts.items()}
        self.global_prior = Counter(labels)
        return self

    def scores(self, texts, vendors):
        out = np.zeros((len(texts), len(CODES)))
        for i, v in enumerate(vendors):
            c = self.table.get(normalize(v), self.global_prior)
            for j, code in enumerate(CODES):
                out[i, j] = c.get(code, 0) + 0.1
        return out / out.sum(axis=1, keepdims=True)


class SupervisedTextMatcher(Matcher):
    needs_training = True

    def __init__(self, use_vendor: bool = True, C: float = 8.0):
        self.use_vendor = use_vendor
        self.name = "char+word LR · history" + (" + vendor" if use_vendor else "")
        self.features = make_union(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\b\w+\b"),
        )
        self.model = LogisticRegression(C=C, max_iter=3000)

    def _prep(self, texts, vendors):
        base = [normalize(t) for t in texts]
        return [with_vendor(t, v) for t, v in zip(base, vendors)] if self.use_vendor else base

    def fit(self, texts, vendors, labels):
        x = self.features.fit_transform(self._prep(texts, vendors))
        self.model.fit(x, labels)
        return self

    def scores(self, texts, vendors):
        proba = self.model.predict_proba(self.features.transform(self._prep(texts, vendors)))
        out = np.zeros((len(texts), len(CODES)))
        for j, cls in enumerate(self.model.classes_):
            out[:, CODES.index(cls)] = proba[:, j]
        return out


class HybridMatcher(Matcher):
    needs_training = True

    def __init__(self, members: list[Matcher], weights: list[float] | None = None):
        self.members = members
        self.weights = weights or [1.0] * len(members)
        self.name = "hybrid · " + " + ".join(m.name.split(" · ")[0] for m in members)

    def fit(self, texts, vendors, labels):
        for m in self.members:
            if m.needs_training:
                m.fit(texts, vendors, labels)
        return self

    def scores(self, texts, vendors):
        log_p = sum(w * np.log(m.scores(texts, vendors) + 1e-9) for m, w in zip(self.members, self.weights))
        return _softmax(log_p, 1.0)


def default_matchers(include_embeddings: bool = False) -> list[Matcher]:
    matchers: list[Matcher] = [
        FuzzyCatalogMatcher(),
        TfidfCatalogMatcher(analyzer="word", ngram_range=(1, 1)),
        TfidfCatalogMatcher(),
    ]
    if include_embeddings:
        matchers.append(EmbeddingCatalogMatcher())
    matchers += [
        VendorPriorMatcher(),
        SupervisedTextMatcher(use_vendor=False),
        SupervisedTextMatcher(use_vendor=True),
        HybridMatcher([SupervisedTextMatcher(use_vendor=True), TfidfCatalogMatcher()], weights=[1.0, 0.3]),
    ]
    return matchers
