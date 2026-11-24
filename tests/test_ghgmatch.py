import numpy as np
import pytest

from ghgmatch.catalog import CATALOG, factor_lookup
from ghgmatch.evaluate import coverage_at_precision, rank_of_truth
from ghgmatch.matchers import CODES, FuzzyCatalogMatcher, HybridMatcher, SupervisedTextMatcher, TfidfCatalogMatcher
from ghgmatch.synth import generate
from ghgmatch.text import normalize


def test_normalize_handles_umlauts_skus_and_quantities():
    assert normalize("Pipettenspitzen 200µl  Stück AB-123456") == "pipettenspitzen stueck"
    assert normalize("Bürostuhl ERGONOMISCH") == "buerostuhl ergonomisch"


def test_catalog_codes_are_unique_and_factors_positive():
    assert len({c.code for c in CATALOG}) == len(CATALOG)
    assert all(v > 0 for v in factor_lookup().values())


def test_generator_is_deterministic():
    a, b = generate(n=50, seed=1), generate(n=50, seed=1)
    assert a.equals(b)
    assert set(a["true_code"]).issubset(CODES)


@pytest.mark.parametrize("matcher", [FuzzyCatalogMatcher(), TfidfCatalogMatcher()])
def test_zero_shot_scores_are_distributions(matcher):
    p = matcher.predict_proba(["Laptop Dell", "Flug Zurich London"], ["Dell", "SWISS"])
    assert p.shape == (2, len(CODES))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-6)


def test_supervised_and_hybrid_learn_the_mapping():
    train, test = generate(n=800, seed=3), generate(n=200, seed=4)
    args = (train["description"].tolist(), train["vendor"].tolist(), train["true_code"].tolist())
    for m in [SupervisedTextMatcher(), HybridMatcher([SupervisedTextMatcher(), TfidfCatalogMatcher()], [1.0, 0.3])]:
        m.fit(*args)
        pred = m.predict_proba(test["description"].tolist(), test["vendor"].tolist()).argmax(axis=1)
        acc = (np.array(CODES)[pred] == test["true_code"].to_numpy()).mean()
        assert acc > 0.85


def test_rank_and_coverage_metrics():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    assert rank_of_truth(proba, np.array([0, 1])).tolist() == [1, 2]
    assert coverage_at_precision(np.array([0.9, 0.8, 0.2]), np.array([True, True, False]), 0.95) == pytest.approx(2 / 3)
