#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce AP-OLR20 Task 2 score-level diagnostics for APTM vs APTM+PGTA.

Input (no header; one line per binary detection trial):
    <enroll_dialect> <utterance_id> <score> <target|nontarget>

EER: pooled binary trials, linearly interpolated ROC crossing (FPR = FNR).
Confusion: target-dialect utterances only, top score across enrolled dialects.
FAR@FRR: model-specific 10th percentile of pooled genuine scores; descriptive
          test-set operating point, NOT a development-set calibrated threshold.

Requires: numpy, scipy, scikit-learn (no training framework required).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def read_scores(path: Path):
    trials = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_no}: expected four fields, got {fields}")
        dialect, utt, value, label = fields
        if label not in ("target", "nontarget"):
            raise ValueError(f"{path}:{line_no}: invalid trial label {label!r}")
        score = float(value)
        if not np.isfinite(score):
            raise ValueError(f"{path}:{line_no}: non-finite score")
        key = (dialect, utt)
        if key in trials:
            raise ValueError(f"{path}:{line_no}: duplicate trial {key}")
        trials[key] = (score, int(label == "target"))

    if not trials:
        raise ValueError(f"{path}: empty score file")
    return trials


def validate_pair(data_a, data_b):
    if set(data_a) != set(data_b):
        a_only = set(data_a) - set(data_b)
        b_only = set(data_b) - set(data_a)
        raise ValueError(f"Trial keys differ: A-only={len(a_only)}, B-only={len(b_only)}")
    mismatches = [k for k in data_a if data_a[k][1] != data_b[k][1]]
    if mismatches:
        raise ValueError(f"Target/nontarget labels differ at {mismatches[:3]}")

    dialects = sorted({d for d, _ in data_a})
    by_utt = defaultdict(dict)
    for (dialect, utt), (_, label) in data_a.items():
        by_utt[utt][dialect] = label
    true_dialect = {}
    for utt, flags in by_utt.items():
        if set(flags) != set(dialects):
            raise ValueError(f"Utterance {utt!r} has incomplete dialect trials")
        positives = [d for d, flag in flags.items() if flag == 1]
        if len(positives) > 1:
            raise ValueError(f"Utterance {utt!r} has multiple target dialects")
        true_dialect[utt] = positives[0] if positives else None
    return dialects, true_dialect


def eer_percent(scores, labels):
    """Return interpolated EER in percent for pooled binary detection trials."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if len(scores) == 0 or not (np.any(labels == 1) and np.any(labels == 0)):
        raise ValueError("EER requires both positive and negative trials")
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    tpr_at_fpr = interp1d(fpr, tpr)
    eer = brentq(lambda x: 1.0 - x - float(tpr_at_fpr(x)), 0.0, 1.0)
    return float(eer * 100.0)


def score_analysis(trials, dialects, truths, target_frr=0.10):
    # Identical trial keys/order are used for both models in the caller.
    utterances = sorted(truths)
    keys = sorted(trials)
    scores = np.array([trials[k][0] for k in keys], dtype=np.float64)
    labels = np.array([trials[k][1] for k in keys], dtype=np.int8)
    is_target_utt = np.array([truths[utt] is not None for _, utt in keys])
    is_non_target_utt = ~is_target_utt

    # Detection EERs are pooled across dialects and source utterances.
    overall_eer = eer_percent(scores, labels)
    target_other_eer = eer_percent(scores[is_target_utt], labels[is_target_utt])
    target_unknown_mask = (labels == 1) | is_non_target_utt
    target_unknown_eer = eer_percent(scores[target_unknown_mask], labels[target_unknown_mask])

    positives = scores[labels == 1]
    target_other_neg = scores[is_target_utt & (labels == 0)]
    unknown_neg = scores[is_non_target_utt]
    # Same target-side FRR setting, but separate threshold per model.
    # Using test labels here is for retrospective diagnostics only.
    threshold = float(np.quantile(positives, target_frr))
    actual_frr = float(100.0 * np.mean(positives < threshold))
    other_far = float(100.0 * np.mean(target_other_neg >= threshold))
    unknown_far = float(100.0 * np.mean(unknown_neg >= threshold))

    # Top-1 confusion is NOT an EER metric. Non-target utterances are excluded.
    matrix = np.zeros((len(dialects), len(dialects)), dtype=int)
    for utt in utterances:
        true = truths[utt]
        if true is None:
            continue
        # Sorted dialect order defines deterministic tie breaking.
        predicted = max(dialects, key=lambda d: trials[(d, utt)][0])
        matrix[dialects.index(true), dialects.index(predicted)] += 1
    total_target_utts = int(matrix.sum())
    confusion_percent = 100.0 * (1.0 - np.trace(matrix) / total_target_utts)

    per_dialect = {}
    for dialect in dialects:
        pos = np.array([trials[(dialect, utt)][0] for utt in utterances
                        if truths[utt] == dialect])
        neg = np.array([trials[(dialect, utt)][0] for utt in utterances
                        if truths[utt] is None])
        per_dialect[dialect] = {
            "target_utterances": len(pos),
            "correct_top1": int(matrix[dialects.index(dialect), dialects.index(dialect)]),
            "top1_accuracy_percent": float(100.0 * matrix[dialects.index(dialect), dialects.index(dialect)] / len(pos)),
            "target_vs_nontarget_eer_percent": eer_percent(
                np.concatenate([pos, neg]),
                np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
            ),
        }

    # Optional diagnostic FAR by non-target source. Language is inferred only
    # from the utterance ID prefix, not from score or assumed numeric indices.
    source_far = {}
    for source in sorted({utt.split("_", 1)[0] for utt in utterances if truths[utt] is None}):
        src_scores = np.array([trials[(d, utt)][0] for utt in utterances
                               if truths[utt] is None and utt.startswith(source + "_")
                               for d in dialects])
        source_far[source] = {
            "negative_trials": len(src_scores),
            "far_percent": float(100.0 * np.mean(src_scores >= threshold)),
        }

    return {
        "n_trials": len(keys),
        "n_utterances": len(utterances),
        "n_target_utterances": sum(v is not None for v in truths.values()),
        "n_nontarget_utterances": sum(v is None for v in truths.values()),
        "overall_eer_percent": overall_eer,
        "target_vs_other_target_eer_percent": target_other_eer,
        "target_vs_nontarget_language_eer_percent": target_unknown_eer,
        "top1_target_confusion_percent": float(confusion_percent),
        "top1_target_errors": int(total_target_utts - np.trace(matrix)),
        "operating_point": {
            "target_frr_requested_percent": float(target_frr * 100),
            "target_frr_actual_percent": actual_frr,
            "score_threshold": threshold,
            "other_target_far_percent": other_far,
            "nontarget_language_far_percent": unknown_far,
        },
        "confusion_matrix": matrix.tolist(),
        "per_dialect": per_dialect,
        "nontarget_source_far": source_far,
    }


def save_confusion_csv(path, dialects, matrix):
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["True / Predicted", *dialects])
        for d, row in zip(dialects, matrix):
            wr.writerow([d, *row])


def fmt_pct(x):
    return f"{x:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aptm", type=Path, default=Path("APTM_task2_test.lst.score"))
    parser.add_argument("--aptm-pgta", type=Path, default=Path("APTM_PGTA_task2_test.lst.score"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_results"))
    parser.add_argument("--frr", type=float, default=0.10,
                        help="Target false rejection operating point (default: 0.10)")
    args = parser.parse_args()
    if not 0.0 < args.frr < 1.0:
        parser.error("--frr must be between 0 and 1")

    data = {"APTM": read_scores(args.aptm),
            "APTM+PGTA": read_scores(args.aptm_pgta)}
    dialects, truths = validate_pair(data["APTM"], data["APTM+PGTA"])
    results = {name: score_analysis(rows, dialects, truths, args.frr)
               for name, rows in data.items()}

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(
        json.dumps({"target_dialects": dialects, "models": results},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, res in results.items():
        file_name = "APTM" if name == "APTM" else "APTM_PGTA"
        save_confusion_csv(out / f"{file_name}_confusion.csv", dialects,
                           res["confusion_matrix"])

    # One row per metric makes comparison and external plotting straightforward.
    metric_names = (
        "overall_eer_percent", "target_vs_other_target_eer_percent",
        "target_vs_nontarget_language_eer_percent", "top1_target_confusion_percent",
    )
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "APTM", "APTM_PGTA", "delta_percentage_points"])
        for m in metric_names:
            a, b = results["APTM"][m], results["APTM+PGTA"][m]
            wr.writerow([m, f"{a:.6f}", f"{b:.6f}", f"{b-a:+.6f}"])
        for m in ("target_frr_actual_percent", "other_target_far_percent",
                  "nontarget_language_far_percent"):
            a, b = (results[k]["operating_point"][m] for k in ("APTM", "APTM+PGTA"))
            wr.writerow([m, f"{a:.6f}", f"{b:.6f}", f"{b-a:+.6f}"])

    print("\nAP-OLR20 Task 2 | matched score diagnostics")
    print(f"Dialects: {', '.join(dialects)}")
    first = results["APTM"]
    print(f"Trials/model: {first['n_trials']}; utterances: {first['n_utterances']}; "
          f"target: {first['n_target_utterances']}; non-target: {first['n_nontarget_utterances']}")
    print(f"{'Metric':45s} {'APTM':>12s} {'APTM+PGTA':>12s}")
    print("-" * 71)
    for title, m in (
        ("Overall pooled EER", "overall_eer_percent"),
        ("Target vs other target dialect pooled EER", "target_vs_other_target_eer_percent"),
        ("Target vs non-target language pooled EER", "target_vs_nontarget_language_eer_percent"),
        ("Target dialect top-1 confusion rate", "top1_target_confusion_percent"),
    ):
        print(f"{title:45s} {fmt_pct(results['APTM'][m]):>12s} "
              f"{fmt_pct(results['APTM+PGTA'][m]):>12s}")
    for title, m in (("Other target FAR @ target FRR", "other_target_far_percent"),
                     ("Non-target FAR @ target FRR", "nontarget_language_far_percent")):
        a, b = [results[k]["operating_point"][m] for k in ("APTM", "APTM+PGTA")]
        print(f"{title+' ~'+fmt_pct(args.frr*100):45s} {fmt_pct(a):>12s} {fmt_pct(b):>12s}")
    print("\nTop-1 target-dialect confusion matrices (rows=true, columns=predicted):")
    for name, res in results.items():
        print(f"  {name}: {res['confusion_matrix']}")
    print("\nPer-dialect target vs non-target language EER (%):")
    for d in dialects:
        a, b = [results[k]["per_dialect"][d]["target_vs_nontarget_eer_percent"]
                for k in ("APTM", "APTM+PGTA")]
        print(f"  {d}: {a:.2f} -> {b:.2f}")
    print(f"\nSaved: {out.resolve()}")
    print("NOTE: top-1 confusion and pooled EER describe different error criteria.")
    print("NOTE: FAR thresholds are selected separately on each model's TEST positives; use DEV for deployment thresholds.")


if __name__ == "__main__":
    main()
