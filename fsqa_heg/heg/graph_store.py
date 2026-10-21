from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_REF_ID_RE = re.compile(r"[^\s\[\]]+\.pdf:P\d+:L\d+")

NODE_TEXT_CHUNK = "text_chunk"
NODE_SPECIES = "species_anchor"
NODE_METRIC = "metric_term"

EDGE_KEYWORD = "keyword_similarity"
EDGE_DOC_NEXT = "doc_next"
EDGE_TABLE_ROW = "table_row_follow"
EDGE_TABLE_HEADER = "table_header_metric"
EDGE_CONTAINS = "contains"
EDGE_CO_PAGE = "co_page"

STRUCTURAL_EDGE_TYPES = frozenset(
    {EDGE_DOC_NEXT, EDGE_TABLE_ROW, EDGE_TABLE_HEADER, EDGE_CONTAINS, EDGE_CO_PAGE}
)

EDGE_TYPE_WEIGHTS: dict[str, float] = {
    EDGE_KEYWORD: 0.25,
    EDGE_DOC_NEXT: 0.55,
    EDGE_TABLE_ROW: 0.70,
    EDGE_TABLE_HEADER: 0.65,
    EDGE_CONTAINS: 0.50,
    EDGE_CO_PAGE: 0.35,
}


def _keywords(text: str, *, min_len: int = 2) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= min_len}
    raw = text or ""
    for seg in _CJK_RE.findall(raw):
        if len(seg) >= min_len:
            words.add(seg)
        for i in range(len(seg) - 1):
            words.add(seg[i : i + 2])
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "is",
        "are",
        "be",
        "as",
        "at",
        "from",
        "by",
        "this",
        "that",
        "it",
    }
    return {w for w in words if w not in stop}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union


def _dist2(p: tuple[float, float], q: tuple[float, float]) -> float:
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


@dataclass(frozen=True)
class RetrievedSubgraph:
    subgraph: dict[str, Any]
    score: float
    kind: str = "seed"
    via: str | None = None


class GraphStore:
    """A lightweight 2D graph memory store (ported from mini-swe-agent)."""

    def __init__(
        self,
        path: Path,
        *,
        similarity_threshold_place: float = 0.18,
        similarity_threshold_link: float = 0.22,
        place_radius: float = 6.0,
        activation_decay: float = 0.98,
        activation_beta: float = 0.35,
        use_spatial_retrieval: bool = True,
        use_structural_expansion: bool = True,
        use_table_heg: bool = True,
    ):
        self.path = Path(path)
        self.similarity_threshold_place = similarity_threshold_place
        self.similarity_threshold_link = similarity_threshold_link
        self.place_radius = place_radius
        self.activation_decay = activation_decay
        self.activation_beta = activation_beta
        self.use_spatial_retrieval = bool(use_spatial_retrieval)
        self.use_structural_expansion = bool(use_structural_expansion)
        self.use_table_heg = bool(use_table_heg)

        self._data: dict[str, Any] = {"subgraphs": [], "edges": [], "meta": {"created_at": time.time()}}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        self._data.setdefault("subgraphs", [])
        self._data.setdefault("edges", [])
        self._data.setdefault("meta", {"created_at": time.time()})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def subgraphs(self) -> list[dict[str, Any]]:
        return self._data["subgraphs"]

    @property
    def edges(self) -> list[dict[str, Any]]:
        return self._data["edges"]

    def decay_activations(self) -> None:
        for sg in self.subgraphs:
            sg["activation"] = float(sg.get("activation", 0.0)) * float(self.activation_decay)

    def _next_id(self, node_type: str = NODE_TEXT_CHUNK) -> str:
        prefix = {
            NODE_TEXT_CHUNK: "sg",
            NODE_SPECIES: "sp",
            NODE_METRIC: "mt",
        }.get(node_type, "nd")
        n = sum(1 for sg in self.subgraphs if str(sg.get("id", "")).startswith(f"{prefix}_"))
        return f"{prefix}_{n + 1:06d}"

    def _occupied_points(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for sg in self.subgraphs:
            coord = sg.get("coord") or {}
            try:
                out.add((int(round(coord.get("x", 0))), int(round(coord.get("y", 0)))))
            except Exception:
                continue
        return out

    def _new_cluster_coord(self) -> tuple[float, float]:
        n = len(self.subgraphs) + 1
        a = n * 0.75
        x = int(round(a * math.cos(a)))
        y = int(round(a * math.sin(a)))
        occupied = self._occupied_points()
        if (x, y) not in occupied:
            return float(x), float(y)
        r = 1
        while r < 1000:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    if (x + dx, y + dy) not in occupied:
                        return float(x + dx), float(y + dy)
            r += 1
        return float(x), float(y)

    def _place_near(self, base: tuple[float, float]) -> tuple[float, float]:
        occupied = self._occupied_points()
        bx, by = int(round(base[0])), int(round(base[1]))
        r = 1
        max_r = int(max(1.0, self.place_radius))
        while r <= max_r:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if dx * dx + dy * dy > r * r:
                        continue
                    p = (bx + dx, by + dy)
                    if p not in occupied:
                        return float(p[0]), float(p[1])
            r += 1
        return self._new_cluster_coord()

    def _node_type(self, sg: dict[str, Any]) -> str:
        return str(sg.get("node_type") or NODE_TEXT_CHUNK)

    def _text_chunk_nodes(self) -> list[dict[str, Any]]:
        return [sg for sg in self.subgraphs if self._node_type(sg) == NODE_TEXT_CHUNK]

    def _best_match(
        self,
        q_keywords: set[str],
        *,
        chunks_only: bool = False,
    ) -> tuple[dict[str, Any] | None, float]:
        pool = self._text_chunk_nodes() if chunks_only else self.subgraphs
        best_sg = None
        best_sim = 0.0
        for sg in pool:
            sim = _jaccard(q_keywords, set(sg.get("keywords", [])))
            if sim > best_sim:
                best_sim = sim
                best_sg = sg
        return best_sg, best_sim

    def _spatial_center(self, qk: set[str]) -> tuple[tuple[float, float] | None, float]:
        best, best_sim = self._best_match(qk, chunks_only=True)
        if best is None or best_sim <= 0:
            return None, 0.0
        return (float(best["coord"]["x"]), float(best["coord"]["y"])), best_sim

    def _place_for_keywords(
        self,
        qk: set[str],
        *,
        parent: dict[str, Any] | None = None,
    ) -> tuple[float, float, float]:
        if parent is not None and parent.get("coord"):
            base_coord = (float(parent["coord"]["x"]), float(parent["coord"]["y"]))
            x, y = self._place_near(base_coord)
            return x, y, 0.0
        best, best_sim = self._best_match(qk, chunks_only=True)
        if best is not None and best_sim >= self.similarity_threshold_place:
            base_coord = (float(best["coord"]["x"]), float(best["coord"]["y"]))
            x, y = self._place_near(base_coord)
        else:
            x, y = self._new_cluster_coord()
        return x, y, float(best_sim)

    def link_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float | None = None,
    ) -> None:
        if not source or not target or source == target:
            return
        for e in self.edges:
            if e.get("source") == source and e.get("target") == target and e.get("type") == edge_type:
                return
        w = float(weight if weight is not None else EDGE_TYPE_WEIGHTS.get(edge_type, 0.5))
        self.edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": w,
            }
        )

    def add_subgraph(
        self,
        *,
        question: str,
        answer: str,
        actions: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
        step_index: int | None = None,
        created_at: float | None = None,
        node_type: str = NODE_TEXT_CHUNK,
        ref_ids: list[str] | None = None,
        page: int | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        self.decay_activations()

        qk = _keywords(question)
        x, y, best_sim = self._place_for_keywords(qk)

        sg_id = self._next_id(node_type)
        sg = {
            "id": sg_id,
            "node_type": node_type,
            "coord": {"x": x, "y": y},
            "activation": min(1.0, float(self.activation_beta) * float(best_sim)),
            "created_at": float(created_at or time.time()),
            "step_index": step_index,
            "question": question,
            "answer": answer,
            "actions": actions or [],
            "observations": observations or [],
            "keywords": sorted(qk),
            "ref_ids": list(ref_ids or []),
            "page": page,
            "line_start": line_start,
            "line_end": line_end,
        }
        self.subgraphs.append(sg)

        self._link_similar(sg, qk)
        self.save()
        return sg

    def add_evidence_node(
        self,
        *,
        node_type: str,
        anchor_text: str,
        ref_ids: list[str] | None = None,
        page: int | None = None,
        line: int | None = None,
        parent: dict[str, Any] | None = None,
        keywords: set[str] | None = None,
    ) -> dict[str, Any]:
        self.decay_activations()
        qk = set(keywords) if keywords else _keywords(anchor_text)
        x, y, best_sim = self._place_for_keywords(qk, parent=parent)
        node = {
            "id": self._next_id(node_type),
            "node_type": node_type,
            "anchor_text": anchor_text,
            "coord": {"x": x, "y": y},
            "activation": min(1.0, float(self.activation_beta) * float(best_sim)),
            "created_at": time.time(),
            "keywords": sorted(qk),
            "ref_ids": list(ref_ids or []),
            "page": page,
            "line": line,
            "question": anchor_text,
            "answer": anchor_text,
            "actions": [],
            "observations": [],
        }
        self.subgraphs.append(node)
        return node

    def _link_similar(self, sg: dict[str, Any], qk: set[str], *, max_links: int = 3) -> None:
        scored: list[tuple[float, dict[str, Any]]] = []
        for other in self.subgraphs:
            if other["id"] == sg["id"]:
                continue
            sim = _jaccard(qk, set(other.get("keywords", [])))
            if sim >= self.similarity_threshold_link:
                scored.append((sim, other))
        scored.sort(key=lambda t: t[0], reverse=True)

        for sim, other in scored[:max_links]:
            self.link_edge(other["id"], sg["id"], EDGE_KEYWORD, weight=float(sim))

    def _hint_boost(self, sg: dict[str, Any], query_hints: dict[str, Any] | None) -> float:
        if not query_hints:
            return 0.0
        boost = 0.0
        anchor = str(sg.get("anchor_text") or "")
        node_type = str(sg.get("node_type") or NODE_TEXT_CHUNK)
        for sp in query_hints.get("species") or []:
            if sp and (sp in anchor or sp in str(sg.get("question", "")) or sp in str(sg.get("answer", ""))):
                boost += 0.85 if node_type == NODE_SPECIES else 0.35
        for mt in query_hints.get("metrics") or []:
            if mt and (mt in anchor or mt in str(sg.get("question", "")) or mt in str(sg.get("answer", ""))):
                boost += 0.65 if node_type == NODE_METRIC else 0.20
        if query_hints.get("has_grade") and node_type == NODE_TEXT_CHUNK:
            if re.search(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ一二三四五六七八九十]+级?", str(sg.get("answer", ""))):
                boost += 0.10
        return boost

    def _node_similarity(self, qk: set[str], sg: dict[str, Any]) -> float:
        sim = _jaccard(qk, set(sg.get("keywords", [])))
        for rid in sg.get("ref_ids") or []:
            rid_terms = _keywords(str(rid))
            if rid_terms and qk:
                overlap = len(qk & rid_terms) / len(qk)
                sim = max(sim, overlap * 0.5)
        return sim

    def _score_seed(
        self,
        qk: set[str],
        sg: dict[str, Any],
        *,
        center: tuple[float, float] | None,
        radius: float,
        query_hints: dict[str, Any] | None,
    ) -> float | None:
        if self._node_type(sg) != NODE_TEXT_CHUNK:
            return None

        sim = self._node_similarity(qk, sg)
        hint = self._hint_boost(sg, query_hints)
        if sim <= 0.0 and hint <= 0.0:
            return None
        sim = max(sim, hint)

        act = float(sg.get("activation", 0.0))
        dist_pen = 0.0
        if self.use_spatial_retrieval and center is not None:
            d2 = _dist2(center, (float(sg["coord"]["x"]), float(sg["coord"]["y"])))
            if d2 > radius * radius:
                return None
            dist_pen = min(1.0, math.sqrt(d2) / max(1e-6, radius))

        score = (0.72 * sim + 0.20 * act) * (1.0 - 0.35 * dist_pen)
        return float(score)

    def _retrieve_bridge_anchors(
        self,
        query: str,
        query_hints: dict[str, Any] | None,
        *,
        max_bridges: int = 4,
    ) -> list[RetrievedSubgraph]:
        """Species/metric nodes as expansion entry points only (not final seeds)."""
        if not self.use_table_heg or not query_hints:
            return []
        qk = _keywords(query)
        if not qk:
            return []

        results: list[RetrievedSubgraph] = []
        for sg in self.subgraphs:
            nt = self._node_type(sg)
            if nt not in {NODE_SPECIES, NODE_METRIC}:
                continue
            hint = self._hint_boost(sg, query_hints)
            if hint <= 0.0:
                continue
            sim = self._node_similarity(qk, sg)
            score = max(hint, sim) + 0.08
            results.append(RetrievedSubgraph(subgraph=sg, score=float(score), kind="bridge"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_bridges]

    def retrieve(
        self,
        query: str,
        *,
        k: int = 4,
        radius: float = 10.0,
        query_hints: dict[str, Any] | None = None,
    ) -> list[RetrievedSubgraph]:
        qk = _keywords(query)
        if not self.subgraphs or not qk:
            return []

        center, _best_sim = self._spatial_center(qk)

        results: list[RetrievedSubgraph] = []
        for sg in self._text_chunk_nodes():
            score = self._score_seed(qk, sg, center=center, radius=radius, query_hints=query_hints)
            if score is None:
                continue
            results.append(RetrievedSubgraph(subgraph=sg, score=float(score), kind="seed"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _edge_adjacency(self) -> dict[str, list[tuple[str, float, str]]]:
        adj: dict[str, list[tuple[str, float, str]]] = {}
        for e in self.edges:
            s = e.get("source")
            t = e.get("target")
            if not isinstance(s, str) or not isinstance(t, str):
                continue
            w = float(e.get("weight", 0.0) or 0.0)
            et = str(e.get("type") or "edge")
            adj.setdefault(s, []).append((t, w, et))
            adj.setdefault(t, []).append((s, w, et))
        return adj

    def expand_neighbors(
        self,
        query: str,
        seeds: Iterable[RetrievedSubgraph],
        *,
        hops: int = 1,
        max_neighbors: int = 8,
        radius: float = 10.0,
        query_hints: dict[str, Any] | None = None,
        bridge_anchors: Iterable[RetrievedSubgraph] | None = None,
    ) -> list[RetrievedSubgraph]:
        if hops <= 0 or max_neighbors <= 0:
            return []

        qk = _keywords(query)
        if not qk or not self.subgraphs:
            return []

        center, _best_sim = self._spatial_center(qk)

        by_id: dict[str, dict[str, Any]] = {str(sg.get("id")): sg for sg in self.subgraphs if isinstance(sg, dict)}
        seed_ids = {str(r.subgraph.get("id")) for r in seeds if isinstance(r.subgraph, dict) and r.subgraph.get("id")}
        seed_scores = {str(r.subgraph.get("id")): float(r.score) for r in seeds if r.subgraph.get("id")}
        for r in bridge_anchors or []:
            if not isinstance(r.subgraph, dict) or not r.subgraph.get("id"):
                continue
            nid = str(r.subgraph["id"])
            seed_ids.add(nid)
            seed_scores[nid] = max(float(r.score), float(seed_scores.get(nid, 0.0)))

        adj = self._edge_adjacency()
        if not adj or not seed_ids:
            return []

        frontier = list(seed_ids)
        seen: set[str] = set(seed_ids)
        best_neighbor: dict[str, RetrievedSubgraph] = {}

        for _depth in range(1, hops + 1):
            next_frontier: list[str] = []
            for u in frontier:
                for v, w, etype in adj.get(u, []):
                    if v in seen:
                        continue
                    structural = etype in STRUCTURAL_EDGE_TYPES
                    if not self.use_structural_expansion and structural:
                        continue
                    seen.add(v)
                    next_frontier.append(v)

                    sg = by_id.get(v)
                    if not sg:
                        continue

                    nt = self._node_type(sg)
                    sim = self._node_similarity(qk, sg)
                    hint = self._hint_boost(sg, query_hints)
                    if nt != NODE_TEXT_CHUNK:
                        if not structural:
                            continue
                    elif not structural and sim <= 0.0 and hint <= 0.0:
                        continue

                    dist_pen = 0.0
                    if self.use_spatial_retrieval and center is not None and nt == NODE_TEXT_CHUNK:
                        d2 = _dist2(center, (float(sg["coord"]["x"]), float(sg["coord"]["y"])))
                        limit = radius * (1.5 if structural else 1.0)
                        if d2 > limit * limit:
                            continue
                        dist_pen = min(1.0, math.sqrt(d2) / max(1e-6, limit))

                    act = float(sg.get("activation", 0.0))
                    base = float(seed_scores.get(u, 0.0))
                    edge_w = float(EDGE_TYPE_WEIGHTS.get(etype, w))
                    semantic = max(sim, hint)
                    if structural and semantic <= 0.0:
                        semantic = 0.15
                    score = (0.40 * base + 0.30 * edge_w + 0.15 * act + 0.15 * semantic) * (
                        1.0 - 0.30 * dist_pen
                    )
                    prev = best_neighbor.get(v)
                    if prev is None or float(score) > float(prev.score):
                        best_neighbor[v] = RetrievedSubgraph(
                            subgraph=sg, score=float(score), kind="neighbor", via=etype
                        )
            frontier = next_frontier

        out = [
            r
            for r in sorted(best_neighbor.values(), key=lambda r: r.score, reverse=True)
            if self._node_type(r.subgraph) == NODE_TEXT_CHUNK
        ]
        return out[:max_neighbors]

    def _direct_ref_ids(self, sg: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        for rid in sg.get("ref_ids") or []:
            refs.add(str(rid))
        for field in ("question", "answer"):
            refs.update(_REF_ID_RE.findall(str(sg.get(field, ""))))
        for obs in sg.get("observations") or []:
            if isinstance(obs, dict):
                for rid in obs.get("ref_ids") or []:
                    refs.add(str(rid))
        return refs

    def collect_ref_ids(
        self,
        retrieved: Iterable[RetrievedSubgraph],
        *,
        bridge_anchors: Iterable[RetrievedSubgraph] | None = None,
    ) -> set[str]:
        """Collect ref_ids; bridge nodes also union linked text_chunk refs."""
        by_id: dict[str, dict[str, Any]] = {str(sg.get("id")): sg for sg in self.subgraphs if isinstance(sg, dict)}
        adj = self._edge_adjacency()
        refs: set[str] = set()
        seen_nodes: set[str] = set()

        def _absorb(node: dict[str, Any]) -> None:
            nid = str(node.get("id") or "")
            if not nid or nid in seen_nodes:
                return
            seen_nodes.add(nid)
            refs.update(self._direct_ref_ids(node))
            if self._node_type(node) == NODE_TEXT_CHUNK:
                return
            for v, _w, etype in adj.get(nid, []):
                if etype not in {EDGE_CONTAINS, EDGE_TABLE_ROW, EDGE_TABLE_HEADER}:
                    continue
                linked = by_id.get(v)
                if linked is not None and self._node_type(linked) == NODE_TEXT_CHUNK:
                    refs.update(self._direct_ref_ids(linked))

        for r in retrieved:
            if isinstance(r.subgraph, dict):
                _absorb(r.subgraph)
        for r in bridge_anchors or []:
            if isinstance(r.subgraph, dict):
                _absorb(r.subgraph)
        return refs

    def _effective_hops(self, hops: int, query_hints: dict[str, Any] | None) -> int:
        if hops <= 0 or not query_hints:
            return hops
        if query_hints.get("is_table_like") or query_hints.get("species"):
            return max(hops, 2)
        return hops

    def query_retrieval(
        self,
        query: str,
        *,
        k: int = 4,
        radius: float = 10.0,
        query_hints: dict[str, Any] | None = None,
        hops: int = 1,
        max_neighbors: int = 8,
        max_bridges: int = 4,
    ) -> tuple[list[RetrievedSubgraph], list[RetrievedSubgraph], list[RetrievedSubgraph]]:
        """Two-stage retrieval: text_chunk seeds + bridge expansion -> text_chunk neighbors."""
        self.decay_activations()
        chunk_seeds = self.retrieve(query, k=k, radius=radius, query_hints=query_hints)
        qk = _keywords(query)
        for r in chunk_seeds:
            sg = r.subgraph
            if self._node_type(sg) != NODE_TEXT_CHUNK:
                continue
            sim = max(self._node_similarity(qk, sg), self._hint_boost(sg, query_hints))
            a = float(sg.get("activation", 0.0))
            sg["activation"] = min(1.0, a + float(self.activation_beta) * float(sim))

        bridges = self._retrieve_bridge_anchors(query, query_hints, max_bridges=max_bridges)
        effective_hops = self._effective_hops(hops, query_hints)
        neighbors: list[RetrievedSubgraph] = []
        if effective_hops > 0 and max_neighbors > 0 and (chunk_seeds or bridges):
            neighbors = self.expand_neighbors(
                query,
                chunk_seeds,
                hops=effective_hops,
                max_neighbors=max_neighbors,
                radius=radius,
                query_hints=query_hints,
                bridge_anchors=bridges,
            )
        self.save()
        return chunk_seeds, bridges, neighbors

    def reinforce(
        self,
        query: str,
        *,
        k: int = 4,
        radius: float = 10.0,
        query_hints: dict[str, Any] | None = None,
    ) -> list[RetrievedSubgraph]:
        seeds, _, _ = self.query_retrieval(
            query,
            k=k,
            radius=radius,
            query_hints=query_hints,
            hops=0,
            max_neighbors=0,
        )
        return seeds

    def to_prompt_snippet(
        self,
        retrieved: Iterable[RetrievedSubgraph],
        *,
        max_chars: int = 1800,
        budget_fraction: float = 1.0,
    ) -> str:
        hard_max = int(max_chars)
        soft_max = int(max(0, min(hard_max, math.floor(hard_max * float(budget_fraction)))))
        if soft_max <= 0:
            soft_max = hard_max

        seeds: list[RetrievedSubgraph] = []
        neighbors: list[RetrievedSubgraph] = []
        for r in retrieved:
            (seeds if r.kind == "seed" else neighbors).append(r)

        lines: list[str] = ["## Graph memory (heterogeneous evidence + 2D proximity)"]

        def _try_add(s: str) -> bool:
            candidate = ("\n".join(lines + [s]).strip() + "\n") if lines else (s.strip() + "\n")
            if len(candidate) <= soft_max:
                lines.append(s)
                return True
            return False

        def _add_entry(r: RetrievedSubgraph, *, q_max: int, a_max: int, show_cmds: bool) -> None:
            sg = r.subgraph
            x, y = sg["coord"]["x"], sg["coord"]["y"]
            node_type = str(sg.get("node_type") or NODE_TEXT_CHUNK)
            prefix = "seed" if r.kind == "seed" else "nbr"
            via = f" via={r.via}" if (r.kind != "seed" and r.via) else ""
            type_tag = f" type={node_type}" if node_type != NODE_TEXT_CHUNK else ""
            _try_add(
                f"- [{prefix}] id={sg['id']} coord=({x:.0f},{y:.0f}) score={r.score:.3f} act={float(sg.get('activation',0)):.2f}{type_tag}{via}"
            )
            if node_type == NODE_SPECIES:
                refs = sg.get("ref_ids") or []
                _try_add(f"  Species: {sg.get('anchor_text', '')} refs={refs[:3]}")
                return
            if node_type == NODE_METRIC:
                refs = sg.get("ref_ids") or []
                _try_add(f"  Metric: {sg.get('anchor_text', '')} refs={refs[:3]}")
                return
            q = _clip(str(sg.get("question", "")), q_max)
            a = _clip(str(sg.get("answer", "")), a_max)
            _try_add(f"  Q: {q}")
            _try_add(f"  A: {a}")
            if not show_cmds:
                return
            cmds: list[str] = []
            for action in sg.get("actions", []) or []:
                c = action.get("command")
                if isinstance(c, str) and c.strip():
                    cmds.append(_clip(c.strip().replace("\n", " "), 120))
            if cmds:
                _try_add(f"  Commands: {', '.join(cmds[:3])}")

        if seeds:
            _try_add("### Seeds")
        for i, r in enumerate(seeds):
            remaining = max(0, soft_max - len("\n".join(lines).strip() + "\n"))
            seeds_left = max(1, len(seeds) - i)
            per = remaining // seeds_left if remaining else 0
            if per < 220:
                q_max, a_max, show_cmds = 120, 140, False
            elif per < 360:
                q_max, a_max, show_cmds = 180, 200, False
            else:
                q_max, a_max, show_cmds = 240, 260, True
            _add_entry(r, q_max=q_max, a_max=a_max, show_cmds=show_cmds)

        if neighbors:
            _try_add("### Neighbors")
        for r in neighbors:
            if len("\n".join(lines).strip() + "\n") >= soft_max:
                break
            _add_entry(r, q_max=200, a_max=220, show_cmds=False)

        out = "\n".join(lines).strip() + "\n"
        return out if len(out) <= hard_max else out[: max(0, hard_max - 1)] + "…"
