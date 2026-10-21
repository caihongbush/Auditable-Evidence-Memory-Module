"""林业标准评测：各实验组的输出路径与 JSON 字段名。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class EvalKeys:
    run_tag: str
    results_suffix: str
    out_file: Path
    pred_key: str
    recall_key: str
    f1_key: str
    cat_key: str
    ctx_key: str
    graph_recall_key: str | None = None
    graph_ctx_key: str | None = None
    lexical_ctx_key: str | None = None


def _norm_suffix(results_suffix: str) -> str:
    s = (results_suffix or "").strip()
    if not s:
        return ""
    return s if s.startswith("_") else f"_{s}"


def resolve_eval_keys(model: str, run_tag: str, results_suffix: str = "") -> EvalKeys:
    tag = (run_tag or "full").strip().lower()
    sf = _norm_suffix(results_suffix)
    if tag == "full":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_graph{sf}_deepseek.json",
            pred_key=f"{model}_standards_graph{sf}_prediction",
            recall_key=f"{model}_standards_recall{sf}",
            f1_key=f"{model}_standards_graph{sf}_f1",
            cat_key=f"{model}_standards_category_prediction{sf}",
            ctx_key=f"{model}_standards_context{sf}",
            graph_recall_key=f"{model}_standards_graph_recall{sf}",
            graph_ctx_key=f"{model}_standards_graph_context{sf}",
            lexical_ctx_key=f"{model}_standards_lexical_context{sf}",
        )
    if tag == "graph_only":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_graph_only{sf}_deepseek.json",
            pred_key=f"{model}_standards_graph_only{sf}_prediction",
            recall_key=f"{model}_standards_graph_only_recall{sf}",
            f1_key=f"{model}_standards_graph_only{sf}_f1",
            cat_key=f"{model}_standards_graph_only_category_prediction{sf}",
            ctx_key=f"{model}_standards_graph_only_context{sf}",
            graph_recall_key=f"{model}_standards_graph_only_recall{sf}",
        )
    if tag == "lexical_only":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_lexical_only{sf}_deepseek.json",
            pred_key=f"{model}_standards_lexical_only{sf}_prediction",
            recall_key=f"{model}_standards_lexical_only_recall{sf}",
            f1_key=f"{model}_standards_lexical_only{sf}_f1",
            cat_key=f"{model}_standards_lexical_only_category_prediction{sf}",
            ctx_key=f"{model}_standards_lexical_only_context{sf}",
            lexical_ctx_key=f"{model}_standards_lexical_only_context{sf}",
        )
    if tag == "dense_rag":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_dense_rag{sf}_deepseek.json",
            pred_key=f"{model}_standards_dense_rag{sf}_prediction",
            recall_key=f"{model}_standards_dense_rag_recall{sf}",
            f1_key=f"{model}_standards_dense_rag{sf}_f1",
            cat_key=f"{model}_standards_dense_rag_category_prediction{sf}",
            ctx_key=f"{model}_standards_dense_rag_context{sf}",
        )
    if tag == "amem":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_amem{sf}_deepseek.json",
            pred_key=f"{model}_amem_prediction{sf}",
            recall_key=f"{model}_amem_recall{sf}",
            f1_key=f"{model}_amem_f1{sf}",
            cat_key=f"{model}_amem_category_prediction{sf}",
            ctx_key=f"{model}_amem_context{sf}",
        )
    if tag == "memgpt":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_memgpt{sf}_deepseek.json",
            pred_key=f"{model}_memgpt_prediction{sf}",
            recall_key=f"{model}_memgpt_recall{sf}",
            f1_key=f"{model}_memgpt_f1{sf}",
            cat_key=f"{model}_memgpt_category_prediction{sf}",
            ctx_key=f"{model}_memgpt_context{sf}",
        )
    if tag == "hipporag":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_hipporag{sf}_deepseek.json",
            pred_key=f"{model}_standards_hipporag{sf}_prediction",
            recall_key=f"{model}_standards_hipporag_recall{sf}",
            f1_key=f"{model}_standards_hipporag{sf}_f1",
            cat_key=f"{model}_standards_hipporag_category_prediction{sf}",
            ctx_key=f"{model}_standards_hipporag_context{sf}",
        )
    if tag == "doc_graphrag":
        return EvalKeys(
            run_tag=tag,
            results_suffix=sf.lstrip("_"),
            out_file=OUTPUTS / f"standards_qa_doc_graphrag{sf}_deepseek.json",
            pred_key=f"{model}_standards_doc_graphrag{sf}_prediction",
            recall_key=f"{model}_standards_doc_graphrag_recall{sf}",
            f1_key=f"{model}_standards_doc_graphrag{sf}_f1",
            cat_key=f"{model}_standards_doc_graphrag_category_prediction{sf}",
            ctx_key=f"{model}_standards_doc_graphrag_context{sf}",
        )
    raise ValueError(
        f"未知 run_tag: {run_tag!r}，可选: full, graph_only, lexical_only, dense_rag, amem, memgpt, hipporag, doc_graphrag"
    )
