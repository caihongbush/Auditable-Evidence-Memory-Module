"""
FSQA 主方法评测算分（full / graph_only / lexical_only）。

用法（在仓库根目录）:
  python -m fsqa_heg.score --run-tag full --results-suffix fsqa1k_llm

与 task_eval/score_standards_qa_ablation.py 逻辑一致（import 路径改为 fsqa_heg.*）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsqa_heg.metrics import f1
from fsqa_heg.eval_keys import resolve_eval_keys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-tag",
        type=str,
        required=True,
        choices=("full", "graph_only", "lexical_only", "dense_rag", "amem", "memgpt", "hipporag", "doc_graphrag"),
    )
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--results", type=str, default="")
    p.add_argument(
        "--results-suffix",
        type=str,
        default="",
        help="与推理时一致，如 ctx4k",
    )
    return p.parse_args()


def score_samples(samples: list[dict], keys, *, write_back: Path) -> dict:
    pred_key = keys.pred_key
    recall_key = keys.recall_key
    f1_key = keys.f1_key
    cat_key = keys.cat_key

    by_cat = defaultdict(lambda: {"n": 0, "recall": 0.0, "f1": 0.0, "cls": 0, "cls_n": 0})
    all_recall: list[float] = []
    all_f1: list[float] = []
    cls_ok = cls_n = 0

    graph_recall_key = keys.graph_recall_key
    graph_recalls: list[float] = []

    for sample in samples:
        for qa in sample.get("qa", []):
            cat = int(qa.get("category", 1))
            by_cat[cat]["n"] += 1
            r = float(qa.get(recall_key, 0))
            by_cat[cat]["recall"] += r
            all_recall.append(r)
            if graph_recall_key and graph_recall_key in qa:
                gr = float(qa[graph_recall_key])
                graph_recalls.append(gr)

            if pred_key in qa:
                sc = round(f1(str(qa[pred_key]), str(qa.get("answer", ""))), 4)
                qa[f1_key] = sc
                by_cat[cat]["f1"] += sc
                all_f1.append(sc)

            if cat_key in qa and qa[cat_key] is not None:
                cls_n += 1
                by_cat[cat]["cls_n"] += 1
                if int(qa[cat_key]) == cat:
                    cls_ok += 1
                    by_cat[cat]["cls"] += 1

    n = len(all_recall) or 1
    metrics = {
        "run_tag": keys.run_tag,
        "results_suffix": keys.results_suffix or None,
        "model": keys.pred_key.split("_")[0],
        "num_samples": len(samples),
        "num_qa": len(all_recall),
        "mean_evidence_recall": sum(all_recall) / n,
        "mean_answer_f1": sum(all_f1) / max(len(all_f1), 1),
        "classification_accuracy": cls_ok / cls_n if cls_n else None,
        "per_category": {},
    }
    if graph_recalls:
        metrics["mean_graph_only_recall"] = sum(graph_recalls) / len(graph_recalls)
    for cat in sorted(by_cat):
        b = by_cat[cat]
        if not b["n"]:
            continue
        metrics["per_category"][str(cat)] = {
            "count": b["n"],
            "mean_recall": b["recall"] / b["n"],
            "mean_f1": b["f1"] / b["n"],
            "classification_accuracy": b["cls"] / b["cls_n"] if b["cls_n"] else None,
        }

    write_back.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path = write_back.with_name(write_back.stem + "_metrics.json")
    stats_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n========== 林业标准 [{keys.run_tag}] ==========")
    print(f"结果: {write_back}")
    print(f"问答条数: {metrics['num_qa']}")
    print(f"检索召回率: {metrics['mean_evidence_recall']:.4f}")
    if metrics.get("mean_graph_only_recall") is not None:
        print(f"检索召回率 (仅图记忆): {metrics['mean_graph_only_recall']:.4f}")
    print(f"问答 F1: {metrics['mean_answer_f1']:.4f}")
    if metrics["classification_accuracy"] is not None:
        print(f"分类准确率: {metrics['classification_accuracy']:.4f}")
    print(f"指标: {stats_path}")
    return metrics


def main() -> None:
    args = parse_args()
    keys = resolve_eval_keys(args.model, args.run_tag, args.results_suffix)
    path = Path(args.results) if args.results else keys.out_file
    if not path.is_file():
        raise SystemExit(f"找不到 {path}，请先跑对应推理脚本。")
    samples = json.loads(path.read_text(encoding="utf-8"))
    score_samples(samples, keys, write_back=path)


if __name__ == "__main__":
    main()
