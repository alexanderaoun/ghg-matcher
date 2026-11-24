from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import catalog_frame, factor_lookup
from .evaluate import benchmark, learning_curve
from .matchers import CODES, HybridMatcher, SupervisedTextMatcher, TfidfCatalogMatcher, default_matchers
from .synth import generate


def cmd_generate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate(n=args.n_train, noise=args.noise, seed=2025).to_csv(out / "procurement_2025_labelled.csv", index=False)
    generate(n=args.n_test, noise=args.noise + 0.1, seed=2026, include_held_out=True).to_csv(out / "procurement_2026.csv", index=False)
    catalog_frame().to_csv(out / "emission_factors.csv", index=False)
    print(f"wrote synthetic data to {out}/")


def cmd_benchmark(args):
    data = Path(args.data)
    train = pd.read_csv(data / "procurement_2025_labelled.csv")
    test = pd.read_csv(data / "procurement_2026.csv")
    results = benchmark(default_matchers(include_embeddings=args.embeddings), train, test)
    results = results.sort_values("top1", ascending=False).reset_index(drop=True)
    curve = learning_curve(
        lambda: HybridMatcher([SupervisedTextMatcher(use_vendor=True), TfidfCatalogMatcher()], weights=[1.0, 0.3]),
        train, test, sizes=[25, 50, 100, 250, 500, 1000, 2000, len(train)],
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "benchmark.csv", index=False)
    curve.to_csv(out / "learning_curve.csv", index=False)
    _write_markdown(results, curve, out / "benchmark.md")
    _plot(results, curve, out)
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(results)
        print(curve)


def cmd_predict(args):
    train = pd.read_csv(args.history)
    lines = pd.read_csv(args.input)
    model = HybridMatcher([SupervisedTextMatcher(use_vendor=True), TfidfCatalogMatcher()], weights=[1.0, 0.3])
    model.fit(train["description"].tolist(), train["vendor"].tolist(), train["true_code"].tolist())
    proba = model.predict_proba(lines["description"].tolist(), lines["vendor"].fillna("").tolist())
    best = proba.argmax(axis=1)
    factors = factor_lookup()
    lines["pred_code"] = [CODES[i] for i in best]
    lines["confidence"] = proba.max(axis=1).round(3)
    lines["alt_code"] = [CODES[i] for i in np.argsort(-proba, axis=1)[:, 1]]
    lines["needs_review"] = lines["confidence"] < args.threshold
    lines["t_co2e"] = (lines["spend_chf"] * lines["pred_code"].map(factors) / 1000).round(4)
    lines.to_csv(args.out, index=False)
    review = int(lines["needs_review"].sum())
    print(f"{len(lines)} lines mapped · {review} flagged for review · {lines['t_co2e'].sum():.1f} t CO2e estimated")


def _write_markdown(results: pd.DataFrame, curve: pd.DataFrame, path: Path):
    cols = {
        "method": "Method", "top1": "Top-1", "top1_seen_items": "Top-1 known items",
        "top1_new_items": "Top-1 new items", "top3": "Top-3", "macro_f1": "Macro-F1",
        "spend_weighted_acc": "Spend-weighted acc.", "emissions_error_pct": "Total CO2e error",
        "auto_coverage_at_99": "Auto-accepted @ 99 % precision", "ms_per_1k": "ms / 1k lines",
    }
    r = results[list(cols)].copy()
    for c in ["top1", "top1_seen_items", "top1_new_items", "top3", "macro_f1", "spend_weighted_acc",
              "auto_coverage_at_99"]:
        r[c] = (100 * r[c]).map("{:.1f} %".format)
    r["emissions_error_pct"] = r["emissions_error_pct"].map("{:+.1f} %".format)
    r["ms_per_1k"] = r["ms_per_1k"].map("{:.0f}".format)
    r = r.rename(columns=cols)
    c = curve.copy()
    c["top1"] = (100 * c["top1"]).map("{:.1f} %".format)
    c["coverage_at_99"] = (100 * c["coverage_at_99"]).map("{:.1f} %".format)
    c.columns = ["Labelled lines", "Top-1", "Auto-accepted @ 99 % precision"]
    path.write_text("## Benchmark\n\n" + r.to_markdown(index=False) + "\n\n## Learning curve (hybrid)\n\n"
                    + c.to_markdown(index=False) + "\n")


def _plot(results: pd.DataFrame, curve: pd.DataFrame, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    r = results.sort_values("top1")
    ax[0].barh(r["method"], 100 * r["top1"], color="#9db4d3", label="Top-1")
    ax[0].barh(r["method"], 100 * r["auto_coverage_at_99"], color="#1f3a5f", height=0.4,
               label="auto-accepted at 99% precision")
    ax[0].set_xlabel("% of 2026 lines")
    ax[0].set_xlim(0, 100)
    ax[0].legend(loc="lower right", frameon=False)
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[1].plot(curve["labelled_lines"], 100 * curve["top1"], marker="o", color="#9db4d3", label="Top-1")
    ax[1].plot(curve["labelled_lines"], 100 * curve["coverage_at_99"], marker="o", color="#1f3a5f",
               label="auto-accepted at 99% precision")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("labelled lines from previous year")
    ax[1].set_ylim(0, 100)
    ax[1].legend(loc="lower right", frameon=False)
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=160)


def main():
    p = argparse.ArgumentParser(prog="ghgmatch", description="Map procurement lines to spend-based emission factors.")
    sub = p.add_subparsers(required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", default="data")
    g.add_argument("--n-train", type=int, default=3000)
    g.add_argument("--n-test", type=int, default=2000)
    g.add_argument("--noise", type=float, default=0.5)
    g.set_defaults(func=cmd_generate)
    b = sub.add_parser("benchmark")
    b.add_argument("--data", default="data")
    b.add_argument("--out", default="results")
    b.add_argument("--embeddings", action="store_true")
    b.set_defaults(func=cmd_benchmark)
    pr = sub.add_parser("predict")
    pr.add_argument("input")
    pr.add_argument("--history", default="data/procurement_2025_labelled.csv")
    pr.add_argument("--out", default="results/predictions.csv")
    pr.add_argument("--threshold", type=float, default=0.8)
    pr.set_defaults(func=cmd_predict)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
