"""Exact-unit binary evaluation with source/target-grouped threshold tuning.

Input: {"methods": [...], "cases": [{"pair_id": ..., "group_keys": [...],
 "complete": true, "units": [{"id": ..., "label": 0|1,
 "scores": {"method": float|null}}]}]}.
Null means explicitly not retrieved, not a failed computation. Such units
are always negative; they remain in the denominator. Scores are not assumed
to be probabilities. Labels must be built separately from detector inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def validate(data):
    methods = data["methods"]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Expected distinct method names")
    seen = set()
    for case in data["cases"]:
        if case["pair_id"] in seen:
            raise ValueError("Duplicate pair ID")
        seen.add(case["pair_id"])
        if case.get("complete") is not True or not case.get("group_keys"):
            raise ValueError("Complete unit census and leakage-group keys required")
        ids = set()
        if not case["units"]:
            raise ValueError("Empty unit census")
        for unit in case["units"]:
            if unit["id"] in ids:
                raise ValueError("Duplicate unit ID")
            ids.add(unit["id"])
            if type(unit["label"]) is not int or unit["label"] not in (0, 1):
                raise ValueError("Labels must be 0 or 1")
            if set(unit["scores"]) != set(methods):
                raise ValueError("Every method must explicitly score or abstain on every unit")
            for score in unit["scores"].values():
                if score is not None and (
                    isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                ):
                    raise ValueError("Nonfinite or invalid score")


def metrics(tp, fp, fn, tn):
    div = lambda a, b: a / b if b else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
        "mcc": (tp * tn - fp * fn) / math.sqrt(denom) if denom else 0.0,
    }


def fit_threshold(cases, method):
    """Exact sweep maximizing pooled (micro) F1 on calibration cases only.

    Convention: score > p. Tied scores enter together. Prefer the higher
    threshold when F1 ties; never turn an unretrieved unit positive.
    """
    units = [u for c in cases for u in c["units"]]
    positives = sum(u["label"] for u in units)
    if not positives or positives == len(units):
        raise ValueError("Threshold calibration requires both classes")
    scored = sorted(
        [(u["scores"][method], u["label"]) for u in units if u["scores"][method] is not None],
        reverse=True,
    )
    if not scored:
        return {"p": 0.0, "calibration_f1": 0.0, "scored_units": 0}
    best = {"p": scored[0][0], "calibration_f1": 0.0, "scored_units": len(scored)}
    tp = fp = i = 0
    while i < len(scored):
        value = scored[i][0]
        while i < len(scored) and scored[i][0] == value:
            tp += scored[i][1]
            fp += 1 - scored[i][1]
            i += 1
        f1 = 2 * tp / (tp + fp + positives)
        if f1 > best["calibration_f1"]:
            best.update(p=math.nextafter(float(value), -math.inf), calibration_f1=f1)
    return best


def grouped_folds(cases, folds=5, seed=2027):
    """Connected components prevent shared source OR target artifacts crossing folds."""
    parents = list(range(len(cases)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, case in enumerate(cases):
        for key in case["group_keys"]:
            if key in owners:
                parents[find(i)] = find(owners[key])
            owners[key] = i
    groups = {}
    for i in range(len(cases)):
        groups.setdefault(find(i), []).append(i)
    if folds < 2 or len(groups) < folds:
        raise ValueError("Insufficient independent groups for requested folds")
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    result = [[] for _ in range(folds)]
    for group in groups:
        slot = min(range(folds), key=lambda j: len(result[j]))
        result[slot].extend(group)
    return result


def evaluate(cases, method, p):
    per_pair = []
    for case in cases:
        tp = fp = fn = tn = 0
        for unit in case["units"]:
            score = unit["scores"][method]
            pred = score is not None and score > p
            label = unit["label"]
            tp += int(pred and label)
            fp += int(pred and not label)
            fn += int(not pred and label)
            tn += int(not pred and not label)
        per_pair.append({"pair_id": case["pair_id"], **metrics(tp, fp, fn, tn)})
    return per_pair


def run(data, folds=5, seed=2027):
    validate(data)
    cases = data["cases"]
    split = grouped_folds(cases, folds, seed)
    result = {
        "protocol": "grouped out-of-fold binary evaluation",
        "objective": "micro-F1 on calibration folds",
        "decision_rule": "score > p; null scores always negative",
        "seed": seed, "pairs": len(cases), "methods": {},
        "folds": [[cases[i]["pair_id"] for i in fold] for fold in split],
    }
    for method in data["methods"]:
        rows, thresholds = [], []
        for k, test_ids in enumerate(split):
            test_ids = set(test_ids)
            train = [c for i, c in enumerate(cases) if i not in test_ids]
            test = [c for i, c in enumerate(cases) if i in test_ids]
            fitted = fit_threshold(train, method)
            thresholds.append({"fold": k, **fitted})
            rows.extend(evaluate(test, method, fitted["p"]))
        total = {key: sum(r[key] for r in rows) for key in ("tp", "fp", "fn", "tn")}
        result["methods"][method] = {
            "micro": metrics(**total),
            "macro": {key: sum(r[key] for r in rows) / len(rows)
                      for key in ("precision", "recall", "f1", "accuracy", "mcc")},
            "fold_thresholds": thresholds,
            "deployment_threshold": fit_threshold(cases, method),
            "deployment_note": "Refit on all cases; NOT an independent test result",
            "per_pair": rows,
        }
    return result


def markdown(result):
    lines = [
        "# Statement binary classification",
        "",
        "Grouped out-of-fold evaluation; each method's threshold maximizes calibration micro-F1.",
        "No Top-K cutoff, no line tolerance. Null/unretrieved units remain negative predictions.",
        "",
        "| Method | Precision | Recall | F1 | Accuracy | MCC | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in result["methods"].items():
        m = row["micro"]
        lines.append(f"| {method} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['f1']:.4f} | {m['accuracy']:.4f} | {m['mcc']:.4f} | "
                     f"{row['macro']['f1']:.4f} |")
    lines += ["", "Deployment thresholds are refit on all data and are not test scores."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(data, args.folds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
