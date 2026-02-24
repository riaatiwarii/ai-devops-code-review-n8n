"""
Evaluate model precision, recall, and latency on a held-out test set.

Usage:
    python evaluate.py \
        --test_data ./dataset/test.jsonl \
        --api_url http://localhost:8000 \
        --output ./eval_results.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iou_match(pred_issues: list, gold_issues: list, severity_filter: str = None) -> tuple[int, int, int]:
    """
    Simple matching: a predicted issue matches gold if same file + severity type.
    Returns (true_positives, false_positives, false_negatives).
    """
    if severity_filter:
        pred_issues = [i for i in pred_issues if i["severity"] == severity_filter]
        gold_issues = [i for i in gold_issues if i["severity"] == severity_filter]

    def key(issue):
        return (issue.get("file", ""), issue.get("type", ""))

    pred_keys = [key(i) for i in pred_issues]
    gold_keys = [key(i) for i in gold_issues]

    tp = sum(1 for k in pred_keys if k in gold_keys)
    fp = len(pred_keys) - tp
    fn = sum(1 for k in gold_keys if k not in pred_keys)
    return tp, fp, fn


def evaluate(test_path: str, api_url: str, output_path: str):
    with open(test_path) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    logger.info(f"Evaluating {len(examples)} examples against {api_url}")

    total_tp = total_fp = total_fn = 0
    high_tp = high_fp = high_fn = 0
    latencies = []
    auto_cleared = 0
    escalated = 0

    for i, ex in enumerate(examples):
        payload = {
            "repo": ex.get("repo", "test/repo"),
            "pr_number": ex.get("pr_number", i),
            "author": "evaluator",
            "language": ex.get("language", "unknown"),
            "diff": ex["diff"],
            "files": [],
            "lines_added": 0,
            "lines_removed": 0,
        }

        start = time.time()
        try:
            resp = requests.post(f"{api_url}/review", json=payload, timeout=120)
            resp.raise_for_status()
            pred = resp.json()
        except Exception as e:
            logger.warning(f"API call failed for example {i}: {e}")
            continue

        latency = (time.time() - start) * 1000
        latencies.append(latency)

        pred_issues = pred.get("issues", [])
        gold_issues = ex["review"].get("issues", [])

        tp, fp, fn = iou_match(pred_issues, gold_issues)
        total_tp += tp; total_fp += fp; total_fn += fn

        htp, hfp, hfn = iou_match(pred_issues, gold_issues, severity_filter="high")
        high_tp += htp; high_fp += hfp; high_fn += hfn

        if not pred_issues:
            auto_cleared += 1
        if any(i["severity"] in ("high", "critical") for i in pred_issues):
            escalated += 1

        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i+1}/{len(examples)}")

    # ── Compute metrics ────────────────────────────────────────────────────────
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    high_precision = high_tp / max(high_tp + high_fp, 1)
    high_recall = high_tp / max(high_tp + high_fn, 1)

    n = len(examples)
    results = {
        "total_examples": n,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "high_severity_precision": round(high_precision, 4),
        "high_severity_recall": round(high_recall, 4),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
        "auto_cleared_ratio": round(auto_cleared / max(n, 1), 4),
        "escalated_ratio": round(escalated / max(n, 1), 4),
    }

    print("\n=== EVALUATION RESULTS ===")
    for k, v in results.items():
        print(f"  {k:<35} {v}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Auto-disable if precision too low (false-positive guard)
    if precision < 0.5:
        logger.warning(
            "⚠️  Precision below 0.5 — consider disabling auto-comment "
            "until model is retrained. Set system_config auto_comment_enabled=false."
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000")
    parser.add_argument("--output", default="./eval_results.json")
    args = parser.parse_args()
    evaluate(args.test_data, args.api_url, args.output)