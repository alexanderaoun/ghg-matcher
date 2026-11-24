from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .catalog import factor_lookup
from .matchers import CODES, Matcher


@dataclass
class Report:
    method: str
    top1: float
    top3: float
    mrr: float
    macro_f1: float
    spend_weighted_acc: float
    emissions_error_pct: float
    auto_coverage_at_99: float
    ms_per_1k: float


def rank_of_truth(proba: np.ndarray, truth_idx: np.ndarray) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    return np.argmax(order == truth_idx[:, None], axis=1) + 1


def coverage_at_precision(confidence: np.ndarray, correct: np.ndarray, target: float = 0.99) -> float:
    order = np.argsort(-confidence)
    running = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(running >= target)[0]
    return float((ok.max() + 1) / len(order)) if len(ok) else 0.0


def emissions(spend: np.ndarray, codes: list[str]) -> np.ndarray:
    f = factor_lookup()
    return spend * np.array([f[c] for c in codes]) / 1000.0


def evaluate(matcher: Matcher, test: pd.DataFrame) -> tuple[Report, np.ndarray]:
    texts, vendors = test["description"].tolist(), test["vendor"].tolist()
    start = time.perf_counter()
    proba = matcher.predict_proba(texts, vendors)
    elapsed = time.perf_counter() - start
    truth = np.array([CODES.index(c) for c in test["true_code"]])
    pred = proba.argmax(axis=1)
    ranks = rank_of_truth(proba, truth)
    correct = pred == truth
    spend = test["spend_chf"].to_numpy()
    pred_codes = [CODES[i] for i in pred]
    true_t = emissions(spend, test["true_code"].tolist()).sum()
    pred_t = emissions(spend, pred_codes).sum()
    report = Report(
        method=matcher.name,
        top1=float(correct.mean()),
        top3=float((ranks <= 3).mean()),
        mrr=float((1.0 / ranks).mean()),
        macro_f1=float(f1_score(truth, pred, average="macro")),
        spend_weighted_acc=float(spend[correct].sum() / spend.sum()),
        emissions_error_pct=float(100 * (pred_t - true_t) / true_t),
        auto_coverage_at_99=coverage_at_precision(proba.max(axis=1), correct),
        ms_per_1k=1000 * elapsed / len(test) * 1000,
    )
    return report, proba


def benchmark(matchers: list[Matcher], train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    novel = test["novel_item"].to_numpy(bool) if "novel_item" in test else np.zeros(len(test), bool)
    for m in matchers:
        if m.needs_training:
            m.fit(train["description"].tolist(), train["vendor"].tolist(), train["true_code"].tolist())
        report, proba = evaluate(m, test)
        truth = np.array([CODES.index(c) for c in test["true_code"]])
        correct = proba.argmax(axis=1) == truth
        row = report.__dict__ | {
            "top1_seen_items": float(correct[~novel].mean()) if (~novel).any() else np.nan,
            "top1_new_items": float(correct[novel].mean()) if novel.any() else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def learning_curve(factory, train: pd.DataFrame, test: pd.DataFrame, sizes: list[int], seeds: int = 3) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)
    rows = []
    for n in sizes:
        for s in range(seeds):
            sample = train.sample(n=min(n, len(train)), random_state=s)
            m = factory().fit(sample["description"].tolist(), sample["vendor"].tolist(), sample["true_code"].tolist())
            r, _ = evaluate(m, test)
            rows.append({"labelled_lines": n, "seed": s, "top1": r.top1, "coverage_at_99": r.auto_coverage_at_99})
    return pd.DataFrame(rows).groupby("labelled_lines", as_index=False)[["top1", "coverage_at_99"]].mean()
