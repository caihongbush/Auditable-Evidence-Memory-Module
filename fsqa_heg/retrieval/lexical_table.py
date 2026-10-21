"""
林业标准文档检索：关键词 + 表格行块扩展（树种名与数值分行问题）。
"""

from __future__ import annotations

import re
from typing import Iterable

REF_ID_RE = re.compile(r"[^\s\[\]]+\.pdf:P\d+:L\d+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_CODE_HINT = re.compile(r"编号|标准号|标准编号")
_CODE_LINE = re.compile(r"DB\d|GB/?T|LY/?T|\/T\s*\d", re.I)
_GRADE_HINT = re.compile(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ一二三四五六七八九十]+级?")
_HEADER_FAMILY = re.compile(r"(科|属|目|纲)$")
_TABLE_FIELD_QUERY = re.compile(r"拉丁学名|适宜区域|适生|树种特性|分布区域|分布情况")
_TABLE_QUERY = re.compile(
    r"(种子|树种|发芽率|净度|含水量|千粒重|优良度|生活力|初植密度|株行距|标准编号|附表|附录)"
)
_SPECIES_NOISE = re.compile(
    r"(的|中|是什么|包括|附表|附录|单位|指南|措施|阶段|卡片|区域是|检查项目|名称是|规则为)"
)
_NUMERIC_LINE = re.compile(r"^[\d\.%％～~\-—\s]+$")

METRIC_TERMS = {
    "净度",
    "发芽率",
    "含水量",
    "千粒重",
    "优良度",
    "生活力",
    "初植密度",
    "株行距",
    "密度",
    "等级",
    "指标",
    "种子",
    "树种",
}
HEADER_FRAGMENTS = {
    "树种",
    "树种名称",
    "净度",
    "发芽率",
    "含水量",
    "千粒重",
    "优良度",
    "生活力",
    "等级",
    "标准",
    "附录",
    "表",
}

SECTION_INTRO_PAGE_MAX = 22
SECTION_PAGE_MARGIN = 2
_LARGE_DOC_LINE_THRESHOLD = 1500
_REF_PAGE_RE = re.compile(r":P(\d+):")


def ref_id_page(ref_id: str) -> int | None:
    m = _REF_PAGE_RE.search(str(ref_id))
    return int(m.group(1)) if m else None


def _merge_page_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted((max(1, lo), hi) for lo, hi in ranges)
    merged: list[tuple[int, int]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        pl, ph = merged[-1]
        if lo <= ph + 1:
            merged[-1] = (pl, max(ph, hi))
        else:
            merged.append((lo, hi))
    return merged


def page_in_ranges(page: int, ranges: list[tuple[int, int]] | None) -> bool:
    if not ranges:
        return True
    return any(lo <= page <= hi for lo, hi in ranges)


def build_species_sections(indexed: list[dict]) -> list[dict]:
    """按文档顺序切分树种锚点块 → 页码区间（用于大文档检索路由）。"""
    anchors: list[dict] = []
    for i, row in enumerate(indexed):
        t = (row.get("text") or "").strip()
        if not _is_species_anchor(t):
            continue
        cjk = "".join(_CJK_RE.findall(t))
        if len(cjk) < 2 or len(cjk) > 10:
            continue
        if t in HEADER_FRAGMENTS or t in METRIC_TERMS:
            continue
        anchors.append(
            {
                "name": t,
                "start_idx": i,
                "page_start": int(row["page"]),
                "line_start": int(row["line"]),
            }
        )
    if not anchors:
        return []

    sections: list[dict] = []
    for j, a in enumerate(anchors):
        end_idx = anchors[j + 1]["start_idx"] - 1 if j + 1 < len(anchors) else len(indexed) - 1
        end_idx = max(a["start_idx"], end_idx)
        sections.append(
            {
                "name": a["name"],
                "page_start": a["page_start"],
                "page_end": int(indexed[end_idx]["page"]),
                "start_idx": a["start_idx"],
                "end_idx": end_idx,
            }
        )
    return sections


def route_section_page_ranges(
    query: str,
    sections: list[dict],
    hints: dict,
    *,
    page_margin: int = SECTION_PAGE_MARGIN,
) -> list[tuple[int, int]] | None:
    """根据问句树种/章节线索返回页码区间；None 表示不限制（全文检索）。"""
    q = query or ""
    ranges: list[tuple[int, int]] = []

    for sp in hints.get("species") or []:
        for sec in sections:
            name = sec["name"]
            if sp == name or sp in name or name in sp:
                lo = max(1, int(sec["page_start"]) - page_margin)
                hi = int(sec["page_end"]) + page_margin
                ranges.append((lo, hi))

    if ranges:
        return _merge_page_ranges(ranges)

    if re.search(r"目的|原则|范围|定义|概述|总则|术语", q):
        return [(1, SECTION_INTRO_PAGE_MAX)]

    return None


def filter_refs_by_page_ranges(
    refs: set[str],
    page_ranges: list[tuple[int, int]] | None,
) -> set[str]:
    if not refs or not page_ranges:
        return refs
    kept = {r for r in refs if page_in_ranges(ref_id_page(r) or -1, page_ranges)}
    return kept if kept else refs


def should_use_section_routing(
    indexed: list[dict], *, min_lines: int = _LARGE_DOC_LINE_THRESHOLD
) -> bool:
    """Whether to enable species-section page routing.

    Historically gated on ``len(indexed) >= min_lines`` (large docs only).
    All documents now use the same section-routing path; ``min_lines`` is kept
    for API compatibility but ignored.
    """
    _ = (indexed, min_lines)
    return True


def large_doc_retrieval_overrides(base: dict) -> dict:
    """检索参数加成（原大文档设置；现对全部文档统一使用）。"""
    out = dict(base)
    out["lexical_k"] = max(int(out.get("lexical_k", 10)), 20)
    out["table_follow"] = max(int(out.get("table_follow", 14)), 24)
    out["retrieve_k"] = max(int(out.get("retrieve_k", 8)), 12)
    out["lexical_window"] = max(int(out.get("lexical_window", 3)), 4)
    return out



def text_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for seg in _CJK_RE.findall(text or ""):
        if len(seg) >= 2:
            terms.add(seg)
        for i in range(len(seg) - 1):
            terms.add(seg[i : i + 2])
    for w in _ALNUM_RE.findall(text or ""):
        if len(w) >= 2:
            terms.add(w.lower())
    return terms


def extract_query_hints(query: str, indexed: list[dict] | None = None) -> dict:
    """从问句抽取树种名、指标等线索。"""
    q = query or ""
    species: list[str] = []

    # 优先：问句子串与索引中的树种锚点行匹配（避免分词成「银杏在」）
    if indexed:
        for row in indexed:
            t = (row.get("text") or "").strip()
            if not _is_species_anchor(t):
                continue
            cjk_len = len("".join(_CJK_RE.findall(t)))
            if cjk_len > 8:
                continue
            if t in q:
                species.append(t)

    for seg in _CJK_RE.findall(q):
        seg = re.sub(r"种子.*$", "", seg)
        seg = re.sub(r"在.*$", "", seg).strip()
        if len(seg) < 2 or len(seg) > 10:
            continue
        if seg in METRIC_TERMS or seg in HEADER_FRAGMENTS:
            continue
        if any(m in q for m in ("要求", "多少", "什么", "如何", "范围", "上限")):
            pass
        if seg in q:
            species.append(seg)

    species = sorted(set(species), key=len, reverse=True)
    filtered: list[str] = []
    for s in species:
        if s in ("广东省", "全省", "国家", "指南", "适宜区域", "拉丁学名"):
            continue
        if _SPECIES_NOISE.search(s):
            continue
        if re.search(r"要求|多少|什么|如何|范围|上限|级", s):
            continue
        if not any(s != t and s in t for t in species):
            filtered.append(s)

    metrics = [t for t in METRIC_TERMS if t in q]
    is_table = bool(_TABLE_QUERY.search(q)) or bool(_TABLE_FIELD_QUERY.search(q))
    return {
        "species": filtered[:3],
        "metrics": metrics,
        "has_grade": bool(_GRADE_HINT.search(q)),
        "is_table_like": is_table,
    }


def _is_numeric_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _NUMERIC_LINE.match(t):
        return True
    if len(t) <= 6 and re.search(r"\d", t):
        return True
    return False


def _is_species_anchor(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 24:
        return False
    if t in HEADER_FRAGMENTS or t in METRIC_TERMS:
        return False
    if _HEADER_FAMILY.search(t):
        return False
    if _is_numeric_line(t):
        return False
    cjk = "".join(_CJK_RE.findall(t))
    if 2 <= len(cjk) <= 10:
        return True
    return False


def _score_line(query: str, row: dict, hints: dict) -> float:
    q_terms = text_terms(query)
    if not q_terms:
        return 0.0
    text = row.get("text", "")
    l_terms = text_terms(text)
    overlap = len(q_terms & l_terms) / len(q_terms)
    if _CODE_HINT.search(query) and _CODE_LINE.search(text):
        overlap += 0.6
    for sp in hints.get("species", []):
        if sp in text:
            overlap += 0.85
    for m in hints.get("metrics", []):
        if m in text or m in query:
            overlap += 0.15
    if hints.get("has_grade") and _GRADE_HINT.search(text):
        overlap += 0.1
    return overlap


def _index_by_page_line(indexed: list[dict]) -> dict[tuple[int, int], int]:
    return {(int(r["page"]), int(r["line"])): i for i, r in enumerate(indexed)}


def _rows_on_page(indexed: list[dict], page: int) -> list[tuple[int, dict]]:
    return [(i, r) for i, r in enumerate(indexed) if int(r["page"]) == page]


def expand_species_table_block(
    indexed: list[dict],
    anchor_idx: int,
    *,
    follow_lines: int = 14,
) -> list[dict]:
    """从树种锚点行起，吞并同页后续表格行（含单独数字格）。"""
    anchor = indexed[anchor_idx]
    page = int(anchor["page"])
    start_line = int(anchor["line"])
    out: list[dict] = []
    for i, row in _rows_on_page(indexed, page):
        ln = int(row["line"])
        if ln < start_line:
            continue
        if ln > start_line + follow_lines:
            break
        if ln > start_line and _is_species_anchor(row["text"]) and row["text"] != anchor["text"]:
            if not _HEADER_FAMILY.search(row["text"].strip()):
                break
        out.append(row)
    return out


def find_species_anchor_indices(indexed: list[dict], species: str) -> list[int]:
    hits: list[int] = []
    for i, row in enumerate(indexed):
        t = row.get("text", "")
        if species in t and _is_species_anchor(t):
            hits.append(i)
        elif species in t and not _is_numeric_line(t):
            hits.append(i)
    return hits


def lexical_retrieve(
    indexed: list[dict],
    query: str,
    *,
    k: int = 12,
    window: int = 2,
    table_follow: int = 14,
    max_rows: int = 120,
    page_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], set[str]]:
    if not indexed:
        return [], set()

    hints = extract_query_hints(query, indexed)

    def _score_all(corpus: list[dict]) -> list[tuple[int, float]]:
        return [(i, _score_line(query, row, hints)) for i, row in enumerate(corpus)]

    def _pick_hits(scored: list[tuple[int, float]], limit: int) -> list[int]:
        scored.sort(key=lambda x: x[1], reverse=True)
        hit_indices: list[int] = []
        for i, sc in scored:
            if sc <= 0 and hit_indices:
                break
            if sc <= 0:
                continue
            hit_indices.append(i)
            if len(hit_indices) >= limit:
                break
        if not hit_indices and scored:
            hit_indices = [scored[0][0]]
        return hit_indices

    corpus = indexed
    index_map = list(range(len(indexed)))
    if page_ranges:
        corpus = []
        index_map = []
        for i, row in enumerate(indexed):
            if page_in_ranges(int(row["page"]), page_ranges):
                corpus.append(row)
                index_map.append(i)

    hit_indices = _pick_hits(_score_all(corpus), k)
    if page_ranges and len(hit_indices) <= 1:
        # 路由区间无有效命中时回退全文
        hit_indices = _pick_hits(_score_all(indexed), k)
        corpus = indexed
        index_map = list(range(len(indexed)))
    else:
        hit_indices = [index_map[i] for i in hit_indices if i < len(index_map)]

    by_page_line: dict[tuple[int, int], dict] = {}

    def _add_rows(rows: Iterable[dict]) -> None:
        for r in rows:
            if len(by_page_line) >= max_rows:
                return
            by_page_line[(int(r["page"]), int(r["line"]))] = r

    # 1) 普通命中 + 窗口扩展
    for idx in hit_indices:
        row = indexed[idx]
        page = int(row["page"])
        line = int(row["line"])
        _add_rows([row])
        for j in range(max(0, idx - window * 4), min(len(indexed), idx + window * 4 + 1)):
            r = indexed[j]
            if int(r["page"]) != page:
                continue
            if abs(int(r["line"]) - line) <= window:
                _add_rows([r])

    # 2) 表格题：按树种名整段拉取同页行块
    if hints.get("is_table_like") or hints.get("species"):
        for sp in hints.get("species", []):
            for aidx in find_species_anchor_indices(indexed, sp):
                _add_rows(expand_species_table_block(indexed, aidx, follow_lines=table_follow))

    # 2b) 页内二次扫描：top 命中页上，按问句相关行扩展邻域
    for idx in hit_indices[: min(8, k)]:
        page = int(indexed[idx]["page"])
        line = int(indexed[idx]["line"])
        for i, r in _rows_on_page(indexed, page):
            if abs(int(r["line"]) - line) <= max(window * 2, 6):
                _add_rows([r])

    # 2c) 定义/原则/附表类：合并 top 命中页的全部行
    if re.search(r"定义|原则|附表\d|附录", query):
        pages = {int(indexed[i]["page"]) for i in hit_indices[:5]}
        for page in pages:
            for i, r in _rows_on_page(indexed, page):
                _add_rows([r])

    # 3) 指标词命中但问句含树种：对高分行再向前找最近树种锚点
    if hints.get("species") and hints.get("metrics"):
        for idx in hit_indices[:5]:
            row = indexed[idx]
            page = int(row["page"])
            line = int(row["line"])
            best_anchor: int | None = None
            best_dist = 999
            for i, r in _rows_on_page(indexed, page):
                if int(r["line"]) > line:
                    continue
                if not _is_species_anchor(r["text"]):
                    continue
                for sp in hints["species"]:
                    if sp in r["text"]:
                        dist = line - int(r["line"])
                        if dist < best_dist:
                            best_dist = dist
                            best_anchor = i
            if best_anchor is not None:
                _add_rows(expand_species_table_block(indexed, best_anchor, follow_lines=table_follow))

    rows = sorted(by_page_line.values(), key=lambda r: (int(r["page"]), int(r["line"])))
    refs = {r["ref_id"] for r in rows}
    return rows, refs


def truncate_text_block(text: str, max_chars: int) -> str:
    """按字符截断检索上下文（用于图记忆 snippet 等纯文本块）。"""
    if max_chars <= 0:
        return ""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def split_proportional_budget(n_lex: int, n_graph: int, total: int) -> tuple[int, int]:
    """按检索结果自然长度比例分配总字符预算。"""
    total = max(0, int(total))
    if total == 0:
        return 0, 0
    n_lex = max(0, int(n_lex))
    n_graph = max(0, int(n_graph))
    if n_graph == 0:
        return total, 0
    if n_lex == 0:
        return 0, total
    lex = int(total * n_lex / (n_lex + n_graph))
    return lex, total - lex


def split_lexical_first_budget(
    total: int,
    *,
    lexical_min_fraction: float = 0.85,
    graph_max_fraction: float = 0.15,
) -> tuple[int, int]:
    """关键词优先：至少 lexical_min_fraction 给关键词，图记忆不超过 graph_max_fraction。"""
    total = max(0, int(total))
    if total == 0:
        return 0, 0
    g_cap = min(total, max(0, int(round(total * graph_max_fraction))))
    lex = total - g_cap
    lex_floor = int(round(total * lexical_min_fraction))
    if lex < lex_floor:
        lex = min(total, lex_floor)
    graph = total - lex
    return lex, graph


def _rows_to_page_blocks(rows: list[dict], *, max_lines_per_block: int = 18) -> list[list[dict]]:
    """同页连续行合并为块（与 format_context_rows 合并规则一致）。"""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: (int(r["page"]), int(r["line"])))
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    cur_page: int | None = None
    for row in ordered:
        page = int(row["page"])
        if cur_page is None:
            cur_page = page
            cur = [row]
        elif page == cur_page and len(cur) < max_lines_per_block:
            cur.append(row)
        else:
            blocks.append(cur)
            cur_page = page
            cur = [row]
    if cur:
        blocks.append(cur)
    return blocks


def _block_relevance_score(query: str, block: list[dict], hints: dict) -> float:
    if not block:
        return 0.0
    return max(_score_line(query, row, hints) for row in block)


def format_context_rows_scored(
    rows: list[dict],
    query: str,
    indexed: list[dict] | None,
    max_chars: int,
) -> str:
    """按与问句相关性从高到低装入上下文，块内仍保持页内行序。"""
    if not rows or max_chars <= 0:
        return "(无)" if not rows else "(无)"
    hints = extract_query_hints(query, indexed)
    blocks = _rows_to_page_blocks(rows)
    ranked = sorted(
        blocks,
        key=lambda b: (
            -_block_relevance_score(query, b, hints),
            int(b[0]["page"]),
            int(b[0]["line"]),
        ),
    )
    parts: list[str] = []
    total = 0
    for block in ranked:
        chunk_parts: list[str] = []
        chunk_len = 0
        for row in block:
            piece = f"[{row['ref_id']}] {row['text']}\n"
            if total + chunk_len + len(piece) > max_chars:
                break
            chunk_parts.append(piece)
            chunk_len += len(piece)
        if chunk_parts:
            parts.extend(chunk_parts)
            total += chunk_len
        if total >= max_chars:
            break
    return "".join(parts) if parts else "(无)"


def truncate_text_block_scored(text: str, query: str, max_chars: int) -> str:
    """图记忆等非结构化文本：按行与问句重叠度优先保留。"""
    if max_chars <= 0:
        return ""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    hints = extract_query_hints(query)
    q_terms = text_terms(query)
    if not q_terms:
        return truncate_text_block(text, max_chars)
    scored: list[tuple[float, int, str]] = []
    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            continue
        lt = text_terms(line)
        sc = len(q_terms & lt) / len(q_terms)
        for sp in hints.get("species", []):
            if sp in line:
                sc += 0.85
        for m in hints.get("metrics", []):
            if m in line:
                sc += 0.15
        scored.append((sc, i, line))
    if not scored:
        return truncate_text_block(text, max_chars)
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    total = 0
    for _, _, line in scored:
        seg = line + "\n"
        if total + len(seg) > max_chars:
            continue
        out.append(seg)
        total += len(seg)
    if out:
        return "".join(out).rstrip()
    return truncate_text_block(text, max_chars)


def pack_full_context_budget(
    lex_rows: list[dict],
    graph_raw: str,
    query: str,
    indexed: list[dict],
    *,
    total_budget: int,
    budget_strategy: str = "lexical_first",
    scored_truncation: bool = True,
    lexical_min_fraction: float = 0.85,
    graph_max_fraction: float = 0.15,
) -> tuple[str, str]:
    """full 模式：在 total_budget 内分配并截断关键词块与图记忆块。"""
    total = max(0, int(total_budget))
    if total == 0:
        return "(无)", "(未检索到相关图记忆)"

    strategy = (budget_strategy or "lexical_first").strip().lower()
    if strategy == "proportional":
        lex_probe = format_context_rows(lex_rows, max_chars=10**7)
        n_lex = len(lex_probe) if lex_probe != "(无)" else 0
        n_graph = len(graph_raw) if graph_raw else 0
        lex_budget, graph_budget = split_proportional_budget(n_lex, n_graph, total)
    else:
        lex_budget, graph_budget = split_lexical_first_budget(
            total,
            lexical_min_fraction=lexical_min_fraction,
            graph_max_fraction=graph_max_fraction,
        )

    if scored_truncation:
        lexical_block = format_context_rows_scored(lex_rows, query, indexed, lex_budget)
        if graph_budget > 0 and graph_raw:
            graph_snippet = truncate_text_block_scored(graph_raw, query, graph_budget)
        else:
            graph_snippet = ""
    else:
        lexical_block = format_context_rows(lex_rows, max_chars=lex_budget) if lex_budget else "(无)"
        graph_snippet = truncate_text_block(graph_raw, graph_budget) if graph_budget and graph_raw else ""

    if not graph_snippet:
        graph_snippet = "(未检索到相关图记忆)"
    return lexical_block, graph_snippet


def format_context_rows(rows: list[dict], max_chars: int = 10000) -> str:
    """输出条文；连续同页行合并为一段便于阅读。"""
    if not rows:
        return "(无)"
    parts: list[str] = []
    total = 0
    buf_page: int | None = None
    buf_lines: list[str] = []

    def flush_buf() -> None:
        nonlocal total, buf_page, buf_lines
        if not buf_lines:
            return
        block = "\n".join(buf_lines) + "\n"
        if total + len(block) > max_chars:
            return
        parts.append(block)
        total += len(block)
        buf_lines = []
        buf_page = None

    for row in rows:
        page = int(row["page"])
        piece = f"[{row['ref_id']}] {row['text']}"
        if buf_page is None:
            buf_page = page
            buf_lines = [piece]
        elif page == buf_page and len(buf_lines) < 18:
            buf_lines.append(piece)
        else:
            flush_buf()
            if total >= max_chars:
                break
            buf_page = page
            buf_lines = [piece]
    flush_buf()
    return "".join(parts) if parts else "(无)"


def iter_graph_ingest_chunks(
    indexed: list[dict],
    *,
    lines_per_turn: int = 5,
    table_follow: int = 12,
    table_aware: bool = True,
) -> list[list[dict]]:
    """图记忆入库：优先按表格行块，否则按固定行数切分。"""
    chunks: list[list[dict]] = []
    i = 0
    while i < len(indexed):
        row = indexed[i]
        if table_aware and _is_species_anchor(row["text"]):
            block = expand_species_table_block(indexed, i, follow_lines=table_follow)
            if block:
                chunks.append(block)
                idxs = [indexed.index(r) for r in block]
                i = max(idxs) + 1
                continue
        n = max(1, lines_per_turn)
        chunks.append(indexed[i : i + n])
        i += n
    return chunks
