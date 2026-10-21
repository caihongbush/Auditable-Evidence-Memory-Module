"""
FSQA 主方法评测：lexical-first 双通道 + HE-G 图记忆 + DeepSeek 生成。

用法（在仓库根目录）:
  set DEEPSEEK_API_KEY=...
  python -m fsqa_heg.evaluate --infer-only --results-suffix fsqa1k_llm
  python -m fsqa_heg.score --run-tag full --results-suffix fsqa1k_llm

与 task_eval/evaluate_standards_qa_graph_memory.py 逻辑一致（import 路径改为 fsqa_heg.*）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from fsqa_heg.llm_client import run_deepseek_chat, run_deepseek_with_examples
from fsqa_heg.heg import GraphMemoryConfig, GraphMemoryRuntime
from fsqa_heg.retrieval.heg_ingest import ingest_standards_hetero_graph
from fsqa_heg.eval_keys import resolve_eval_keys
from fsqa_heg.score import score_samples as score_standards_samples
from fsqa_heg.retrieval.lexical_table import (
    build_species_sections,
    extract_query_hints,
    filter_refs_by_page_ranges,
    format_context_rows,
    format_context_rows_scored,
    large_doc_retrieval_overrides,
    lexical_retrieve,
    pack_full_context_budget,
    route_section_page_ranges,
    should_use_section_routing,
    truncate_text_block,
    truncate_text_block_scored,
)

ROOT = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


DEFAULT_DATA = ROOT / "data" / "standards_qa.json"
DEFAULT_INDEX_DIR = ROOT / "data" / "standards_indexed"
DEFAULT_GRAPH_DIR = ROOT / "data" / "graph_memory_standards_runs"
DEFAULT_OUT = ROOT / "outputs" / "standards_qa_graph_deepseek.json"

REF_ID_RE = re.compile(r"[^\s\[\]]+\.pdf:P\d+:L\d+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_CODE_HINT = re.compile(r"编号|标准号|标准编号")
_CODE_LINE = re.compile(r"DB\d|GB/?T|LY/?T|\/T\s*\d", re.I)

STANDARDS_QA_PROMPT = """以下是从本标准文档中检索到的条文（含位置标记 [文件名:P页:L行]）。请仅根据下列内容作答。

### 关键词检索（同文档，优先参考）
{lexical_block}

### 图记忆检索
{graph_snippet}

请根据上述条文，用简短短语回答问题；尽量使用原文表述。若条文中有答案，直接写出，勿回答「未提及」。

Question: {question}
Short answer:"""

LEXICAL_ONLY_QA_PROMPT = """以下是从本标准文档中通过关键词与表格行块检索得到的条文（含位置标记 [文件名:P页:L行]）。请仅根据下列内容作答。

### 关键词检索
{lexical_block}

请根据上述条文，用简短短语回答问题；尽量使用原文表述。若条文中有答案，直接写出，勿回答「未提及」。

Question: {question}
Short answer:"""

GRAPH_ONLY_QA_PROMPT = """以下是从本标准文档中通过图记忆检索得到的条文（含位置标记 [文件名:P页:L行]）。请仅根据下列内容作答。

### 图记忆检索
{graph_snippet}

请根据上述条文，用简短短语回答问题；尽量使用原文表述。若条文中有答案，直接写出，勿回答「未提及」。

Question: {question}
Short answer:"""

CATEGORY_PROMPT = """根据下面的林业标准相关问题，判断其知识类型，只输出 JSON：{{"category": 1}} 或 2 或 3。
1=事实/定义/参数，2=程序/要求/步骤/判定规则，3=综合判断或多处综合。

Question: {question}
"""

CATEGORY_EXAMPLES = [
    (
        "种子净度的定义是什么？",
        '{"category": 1}',
    ),
    (
        "含水量达不到要求的种子如何判定？",
        '{"category": 2}',
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--out-file", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--graph-mem-dir", type=str, default=str(DEFAULT_GRAPH_DIR))
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--deepseek-base-url", type=str, default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--skip-classification", action="store_true", help="不调用 LLM 做 category 预测")
    p.add_argument(
        "--infer-only",
        action="store_true",
        help="只跑图记忆检索+生成答案，不计算 F1、不输出汇总指标（之后用 score_standards_qa_graph_memory.py）",
    )
    p.add_argument("--ingest-lines-per-turn", type=int, default=5, help="图记忆非表格段落的合并行数")
    p.add_argument("--table-follow", type=int, default=14, help="命中树种行后同页向后扩展行数（含数字格）")
    p.add_argument("--retrieve-k", type=int, default=8)
    p.add_argument("--inject-max-chars", type=int, default=4000)
    p.add_argument("--lexical-k", type=int, default=10, help="关键词检索种子命中条数")
    p.add_argument("--lexical-window", type=int, default=3, help="普通命中行同页上下扩展行数")
    p.add_argument("--no-lexical", action="store_true", help="仅用图记忆，不做关键词补充")
    p.add_argument("--lexical-only", action="store_true", help="仅关键词检索，不建图、不用图记忆")
    p.add_argument(
        "--run-tag",
        type=str,
        default="full",
        choices=("full", "graph_only", "lexical_only"),
        help="实验组：full=图+关键词；graph_only=仅图；lexical_only=仅关键词",
    )
    p.add_argument(
        "--context-budget-chars",
        type=int,
        default=0,
        help="检索进 prompt 的总字符预算；0=沿用旧设定（关键词 10000，full 另加图 4000）。"
        "lexical_only 用满预算；full 按两路自然长度比例拆分。",
    )
    p.add_argument(
        "--results-suffix",
        type=str,
        default="",
        help="结果文件名/JSON 字段后缀，如 ctx4k → standards_qa_graph_ctx4k_deepseek.json，不覆盖旧结果。",
    )
    p.add_argument(
        "--budget-strategy",
        type=str,
        default="proportional",
        choices=("proportional", "lexical_first"),
        help="full + context-budget 时如何拆分关键词/图记忆预算；lexical_first=关键词至少约 85%%。",
    )
    p.add_argument(
        "--scored-truncation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按问句相关性优先保留条文/图记忆行（默认开启）。",
    )
    p.add_argument(
        "--lexical-budget-fraction",
        type=float,
        default=0.85,
        help="lexical_first 时关键词最低占比（其余给图，图另受 graph-max-fraction 限制）。",
    )
    p.add_argument(
        "--graph-max-fraction",
        type=float,
        default=0.15,
        help="lexical_first 时图记忆最多占总预算比例。",
    )
    p.add_argument(
        "--no-spatial",
        action="store_true",
        help="消融：检索阶段关闭二维半径过滤与距离惩罚（构图仍保留坐标；建议 --results-suffix no_spatial）。",
    )
    p.add_argument(
        "--no-expand",
        action="store_true",
        help="消融：仅种子检索，关闭一跳邻域扩散（neighbor_hops=0；建议 --results-suffix no_expand）。",
    )
    p.add_argument(
        "--no-structural-expand",
        action="store_true",
        help="消融：邻域扩散时不走结构边（doc_next/table_row_follow 等），仅 keyword_similarity（建议 --results-suffix no_structural）。",
    )
    p.add_argument(
        "--no-table-heg",
        action="store_true",
        help="消融：HE-G 关闭表格感知（固定行切块、无树种/指标桥接、无 table 边；建议 --results-suffix no_table_heg）。",
    )
    p.add_argument(
        "--section-aware",
        action="store_true",
        help="兼容旧开关：章节路由 + 检索参数加成现已对全部文档默认启用（无需再传）。",
    )
    p.add_argument(
        "--large-doc-min-lines",
        type=int,
        default=1500,
        help="兼容旧参数：行数阈值已不再用于分流（全部文档统一走章节路由）。",
    )
    p.add_argument(
        "--only-sample-id",
        type=str,
        default="",
        help="仅评测指定 sample_id（用于大文档增量重跑）。",
    )
    p.add_argument(
        "--merge-from",
        type=str,
        default="",
        help="与 --only-sample-id 联用：从已有结果 JSON 合并其它 sample。",
    )
    p.add_argument(
        "--rerun-zero-recall-from",
        type=str,
        default="",
        help="仅重跑该结果文件中 recall=0 的题目（需 --merge-from 指向同一全量结果）。",
    )
    p.add_argument(
        "--exclude-sample-id",
        action="append",
        default=[],
        help="跳过指定 sample（可重复传入）。",
    )
    p.add_argument(
        "--skip-orig302",
        action="store_true",
        help="与 --rerun-zero-recall-from 联用：不重跑初始 302 题中的 recall=0。",
    )
    return p.parse_args()


def load_indexed_lines(sample_id: str, index_dir: Path) -> list[dict]:
    path = index_dir / f"{sample_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少索引文件: {path}，请先运行 scripts/extract_standards_qa.py --dry-run")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["indexed_lines"]


def ingest_standards_sample(
    runtime: GraphMemoryRuntime,
    indexed: list[dict],
    *,
    lines_per_turn: int,
    table_follow: int,
    use_table_heg: bool = True,
) -> None:
    ingest_standards_hetero_graph(
        runtime.store,
        indexed,
        lines_per_turn=lines_per_turn,
        table_follow=table_follow,
        use_table_heg=use_table_heg,
    )


_RAW_SNIPPET_CHARS = 2_000_000


def build_graph_snippet(
    runtime: GraphMemoryRuntime,
    query: str,
    *,
    max_chars: int,
    query_hints: dict | None = None,
) -> str:
    """构建图记忆 prompt 片段；max_chars 控制注入长度。"""
    mem = runtime.config
    seeds, _bridges, neighbors = runtime.store.query_retrieval(
        query,
        k=int(mem.retrieve_k),
        radius=float(mem.retrieve_radius),
        query_hints=query_hints,
        hops=int(mem.neighbor_hops),
        max_neighbors=int(mem.neighbor_k),
    )
    if not seeds and not neighbors:
        return ""
    text = runtime.store.to_prompt_snippet(
        list(seeds) + list(neighbors),
        max_chars=int(max_chars),
        budget_fraction=float(mem.inject_budget_fraction),
    )
    return text.strip() if text and text.strip() else ""


def retrieved_ref_ids(
    runtime: GraphMemoryRuntime,
    query: str,
    *,
    k: int,
    query_hints: dict | None = None,
) -> set[str]:
    mem = runtime.config
    seeds, bridges, neighbors = runtime.store.query_retrieval(
        query,
        k=k,
        radius=float(mem.retrieve_radius),
        query_hints=query_hints,
        hops=int(mem.neighbor_hops),
        max_neighbors=int(mem.neighbor_k),
    )
    return runtime.store.collect_ref_ids(list(seeds) + list(neighbors), bridge_anchors=bridges)


def evidence_recall_score(gold_evidence: list[str], retrieved: set[str]) -> float:
    if not gold_evidence:
        return 1.0
    hits = sum(1 for ev in gold_evidence if ev in retrieved)
    return hits / len(gold_evidence)


def parse_category_json(text: str) -> int | None:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
        return int(obj.get("category"))
    except (json.JSONDecodeError, TypeError, ValueError):
        m = re.search(r'"category"\s*:\s*([123])', text)
        if m:
            return int(m.group(1))
    return None


def predict_category(question: str, args: argparse.Namespace) -> int | None:
    try:
        raw = run_deepseek_with_examples(
            "你只输出 JSON，字段 category 为整数 1、2 或 3。",
            CATEGORY_EXAMPLES,
            CATEGORY_PROMPT.format(question=question),
            model=args.model,
            base_url=args.deepseek_base_url,
            max_tokens=64,
            temperature=0.0,
            timeout=args.request_timeout,
        )
        return parse_category_json(raw)
    except Exception as e:
        print(f"  [分类失败] {e}")
        return None


def save_results(path: str, in_samples: list[dict], out_by_id: dict[str, dict]) -> None:
    ordered = [out_by_id[s["sample_id"]] for s in in_samples if s["sample_id"] in out_by_id]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def print_metrics(samples: list[dict], keys, *, write_back: Path) -> dict:
    """汇总指标；keys 来自 resolve_eval_keys，与 JSON 字段名一致。"""
    return score_standards_samples(samples, keys, write_back=write_back)


def main() -> None:
    args = parse_args()
    _load_local_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("请设置 DEEPSEEK_API_KEY")

    run_tag = args.run_tag
    if run_tag == "graph_only":
        args.no_lexical = True
    if run_tag == "lexical_only":
        args.lexical_only = True

    ablation_flags = ""
    if args.no_spatial:
        ablation_flags += "  no_spatial=1"
    if args.no_expand:
        ablation_flags += "  no_expand=1"
    if args.no_structural_expand:
        ablation_flags += "  no_structural=1"
    if args.no_table_heg:
        ablation_flags += "  no_table_heg=1"

    keys = resolve_eval_keys(args.model, run_tag, args.results_suffix)
    ctx_budget = int(args.context_budget_chars)
    if ctx_budget > 0:
        extra = ""
        if run_tag == "full" and not args.lexical_only and not args.no_lexical:
            extra = (
                f"  budget={args.budget_strategy}"
                f"  scored_truncation={args.scored_truncation}"
            )
        print(
            f"[实验组] {keys.run_tag} (suffix={keys.results_suffix or '—'}) "
            f"-> {keys.out_file.name}  context_budget={ctx_budget}{extra}{ablation_flags}"
        )
    else:
        print(f"[实验组] {keys.run_tag} -> {keys.out_file.name}{ablation_flags}")

    samples = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    index_dir = Path(args.index_dir)
    graph_dir = Path(args.graph_mem_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    pred_key = keys.pred_key
    recall_key = keys.recall_key
    graph_recall_key = keys.graph_recall_key or ""
    cat_key = keys.cat_key
    ctx_key = keys.ctx_key
    graph_ctx_key = keys.graph_ctx_key or ""
    lexical_ctx_key = keys.lexical_ctx_key or ""

    out_path = Path(args.out_file) if args.out_file != str(DEFAULT_OUT) else keys.out_file
    merge_from = Path(args.merge_from) if args.merge_from else None
    if merge_from and merge_from.is_file():
        out_samples = {s["sample_id"]: s for s in json.loads(merge_from.read_text(encoding="utf-8"))}
    elif out_path.is_file() and not args.overwrite:
        out_samples = {s["sample_id"]: s for s in json.loads(out_path.read_text(encoding="utf-8"))}
    else:
        out_samples = {}

    only_sid = (args.only_sample_id or "").strip()
    exclude_sids = set(args.exclude_sample_id or [])
    if only_sid and not (merge_from and merge_from.is_file()):
        raise SystemExit("使用 --only-sample-id 时必须指定 --merge-from 指向已有全量结果 JSON")

    rerun_miss_path = Path(args.rerun_zero_recall_from) if args.rerun_zero_recall_from else None
    rerun_keys: set[tuple[str, str]] | None = None
    if rerun_miss_path:
        if not merge_from or not merge_from.is_file():
            raise SystemExit("使用 --rerun-zero-recall-from 时必须指定 --merge-from")
        src = json.loads(rerun_miss_path.read_text(encoding="utf-8"))
        rerun_keys = set()
        for s in src:
            sid0 = s["sample_id"]
            if sid0 in exclude_sids:
                continue
            for q in s["qa"]:
                if float(q.get(recall_key, 0) or 0) < 0.001:
                    rerun_keys.add((sid0, q["question"]))
        print(f"[重跑 recall=0] {len(rerun_keys)} 题")
        if args.skip_orig302:
            orig302_path = Path(args.data_file) if args.data_file != str(DEFAULT_DATA) else DEFAULT_DATA
            orig302_path = Path(orig302_path).parent / "standards_qa_302_backup.json"
            if orig302_path.is_file():
                orig_keys = {
                    (s["sample_id"], q["question"])
                    for s in json.loads(orig302_path.read_text(encoding="utf-8"))
                    for q in s["qa"]
                }
                rerun_keys = {k for k in rerun_keys if k not in orig_keys}
                print(f"[跳过 orig302] 剩余 {len(rerun_keys)} 题")

    for in_data in tqdm(samples, desc="standards"):
        sid = in_data["sample_id"]
        if sid in exclude_sids:
            continue
        if only_sid and sid != only_sid:
            continue
        if rerun_keys is not None:
            if not any((sid, q["question"]) in rerun_keys for q in in_data["qa"]):
                continue
        elif (
            sid in out_samples
            and not args.overwrite
            and all(pred_key in q for q in out_samples[sid].get("qa", []))
        ):
            print(f"[跳过已完成] {sid}")
            continue

        try:
            indexed = load_indexed_lines(sid, index_dir)
        except FileNotFoundError as e:
            print(f"[跳过] {sid}: {e}")
            continue

        # Unified pipeline: all docs use section routing + large-doc retrieval overrides.
        use_sections = should_use_section_routing(
            indexed, min_lines=int(args.large_doc_min_lines)
        )
        sections = build_species_sections(indexed) if use_sections else []
        if use_sections:
            print(f"[章节路由] {sid}: {len(sections)} 个树种块, {len(indexed)} 行")

        retr = {
            "lexical_k": int(args.lexical_k),
            "table_follow": int(args.table_follow),
            "retrieve_k": int(args.retrieve_k),
            "lexical_window": int(args.lexical_window),
        }
        retr = large_doc_retrieval_overrides(retr)

        runtime = None
        if not args.lexical_only:
            store_path = graph_dir / f"{sid}_graph_memory.json"
            if store_path.exists():
                store_path.unlink()
            mem_cfg = GraphMemoryConfig(
                enabled=True,
                store_path=store_path,
                retrieve_k=int(retr["retrieve_k"]),
                inject_max_chars=int(args.inject_max_chars),
                use_spatial_retrieval=not bool(args.no_spatial),
                use_structural_expansion=not bool(args.no_structural_expand),
                use_table_heg=not bool(args.no_table_heg),
                neighbor_hops=0 if args.no_expand else 2,
                neighbor_k=16,
            )
            runtime = GraphMemoryRuntime(mem_cfg)
            ingest_standards_sample(
                runtime,
                indexed,
                lines_per_turn=int(args.ingest_lines_per_turn),
                table_follow=int(retr["table_follow"]),
                use_table_heg=not bool(args.no_table_heg),
            )

        if sid in out_samples:
            out_data = {"sample_id": sid, "source_document": in_data.get("source_document"), "qa": []}
            for q in out_samples[sid]["qa"]:
                out_data["qa"].append(dict(q))
        else:
            out_data = {
                "sample_id": sid,
                "source_document": in_data.get("source_document"),
                "qa": [dict(q) for q in in_data["qa"]],
            }

        completed = 0
        for i, qa in enumerate(tqdm(out_data["qa"], desc=f"qa {sid}", leave=False)):
            q_text = qa["question"]
            if rerun_keys is not None:
                if (sid, q_text) not in rerun_keys:
                    continue
            elif pred_key in qa and not args.overwrite:
                continue

            hints = extract_query_hints(q_text, indexed)
            page_ranges = (
                route_section_page_ranges(q_text, sections, hints) if use_sections else None
            )
            graph_refs: set[str] = set()
            snippet = ""
            graph_raw = ""
            if runtime is not None:
                raw_cap = _RAW_SNIPPET_CHARS if ctx_budget > 0 else int(args.inject_max_chars)
                graph_raw = build_graph_snippet(
                    runtime, q_text, max_chars=raw_cap, query_hints=hints
                )
                snippet = graph_raw
                graph_refs = retrieved_ref_ids(
                    runtime, q_text, k=int(retr["retrieve_k"]), query_hints=hints
                )
                if page_ranges:
                    graph_refs = filter_refs_by_page_ranges(graph_refs, page_ranges)

            if args.lexical_only or args.no_lexical:
                use_lexical = args.lexical_only
            else:
                use_lexical = True
            if use_lexical:
                lex_rows, lex_refs = lexical_retrieve(
                    indexed,
                    q_text,
                    k=int(retr["lexical_k"]),
                    window=int(retr["lexical_window"]),
                    table_follow=int(retr["table_follow"]),
                    page_ranges=page_ranges,
                )
            else:
                lex_rows, lex_refs = [], set()

            combined_refs = graph_refs | lex_refs
            graph_recall = evidence_recall_score(qa.get("evidence") or [], graph_refs)
            if args.lexical_only:
                recall = evidence_recall_score(qa.get("evidence") or [], lex_refs)
            elif args.no_lexical:
                recall = graph_recall
            else:
                recall = evidence_recall_score(qa.get("evidence") or [], combined_refs)

            if ctx_budget > 0:
                use_scored = bool(args.scored_truncation)
                if args.lexical_only:
                    if use_scored:
                        lexical_block = format_context_rows_scored(
                            lex_rows, q_text, indexed, ctx_budget
                        )
                    else:
                        lexical_block = format_context_rows(lex_rows, max_chars=ctx_budget)
                    graph_snippet = ""
                elif args.no_lexical:
                    lexical_block = ""
                    if use_scored and graph_raw:
                        graph_snippet = truncate_text_block_scored(graph_raw, q_text, ctx_budget)
                    else:
                        graph_snippet = (
                            truncate_text_block(graph_raw, ctx_budget)
                            if graph_raw
                            else "(未检索到相关图记忆)"
                        )
                    snippet = graph_snippet
                else:
                    lexical_block, graph_snippet = pack_full_context_budget(
                        lex_rows,
                        graph_raw,
                        q_text,
                        indexed,
                        total_budget=ctx_budget,
                        budget_strategy=args.budget_strategy,
                        scored_truncation=use_scored,
                        lexical_min_fraction=float(args.lexical_budget_fraction),
                        graph_max_fraction=float(args.graph_max_fraction),
                    )
                    snippet = graph_snippet
            else:
                lexical_block = format_context_rows(lex_rows)
                graph_snippet = snippet if snippet else "(未检索到相关图记忆)"

            if args.lexical_only:
                prompt = LEXICAL_ONLY_QA_PROMPT.format(
                    lexical_block=lexical_block,
                    question=q_text,
                )
            elif args.no_lexical:
                prompt = GRAPH_ONLY_QA_PROMPT.format(
                    graph_snippet=graph_snippet if graph_snippet else "(未检索到相关图记忆)",
                    question=q_text,
                )
            else:
                prompt = STANDARDS_QA_PROMPT.format(
                    lexical_block=lexical_block,
                    graph_snippet=graph_snippet if graph_snippet else "(未检索到相关图记忆)",
                    question=q_text,
                )
            try:
                answer = run_deepseek_chat(
                    [{"role": "user", "content": prompt}],
                    model=args.model,
                    api_key=api_key,
                    base_url=args.deepseek_base_url,
                    max_tokens=int(args.max_output_tokens),
                    timeout=int(args.request_timeout),
                )
            except Exception as e:
                print(f"  [生成失败] {q_text[:40]}... {e}")
                answer = ""

            qa[pred_key] = answer.strip()
            qa[recall_key] = round(recall, 4)
            qa[ctx_key] = sorted(combined_refs if not args.lexical_only else lex_refs)
            if graph_recall_key:
                qa[graph_recall_key] = round(graph_recall, 4)
            if graph_ctx_key:
                qa[graph_ctx_key] = sorted(graph_refs)
            if lexical_ctx_key:
                qa[lexical_ctx_key] = sorted(lex_refs)

            if not args.skip_classification:
                pred_cat = predict_category(q_text, args)
                qa[cat_key] = pred_cat

            completed += 1
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                out_samples[sid] = out_data
                save_results(str(out_path), samples, out_samples)

        out_samples[sid] = out_data
        save_results(str(out_path), samples, out_samples)

    print(f"\n推理完成: {out_path}")
    if args.infer_only:
        print("未计算指标。请运行:")
        sfx = f" --results-suffix {keys.results_suffix}" if keys.results_suffix else ""
        print(f"  python -m fsqa_heg.score --run-tag {keys.run_tag} --model {args.model}{sfx}")
        return

    final = [out_samples[s["sample_id"]] for s in samples if s["sample_id"] in out_samples]
    print_metrics(final, keys, write_back=out_path)
    print("（汇总 F1 也可用 score_standards_qa_ablation.py 单独重算）")


if __name__ == "__main__":
    main()
