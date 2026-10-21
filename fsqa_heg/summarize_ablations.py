"""Print a small table of pathway ablation metrics from outputs/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-suffix", default="fsqa_ablation")
    args = p.parse_args()
    out = Path("outputs")
    rows = [
        ("lexical_only", f"standards_qa_lexical_only_{args.results_suffix}_deepseek_metrics.json"),
        ("graph_only", f"standards_qa_graph_only_{args.results_suffix}_deepseek_metrics.json"),
        ("full", f"standards_qa_graph_{args.results_suffix}_deepseek_metrics.json"),
    ]
    print(f"{'tag':<14} {'Recall':>8} {'F1':>8}")
    print("-" * 32)
    for tag, fname in rows:
        m = load_metrics(out / fname)
        if m is None:
            print(f"{tag:<14} {'—':>8} {'—':>8}")
        else:
            print(f"{tag:<14} {m.get('mean_evidence_recall', 0):>8.3f} {m.get('mean_answer_f1', 0):>8.3f}")


if __name__ == "__main__":
    main()
