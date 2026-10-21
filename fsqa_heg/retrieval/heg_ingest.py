"""Build heterogeneous evidence graph from indexed standard-document lines."""

from __future__ import annotations

from typing import Any

from fsqa_heg.heg.graph_store import (
    EDGE_CONTAINS,
    EDGE_CO_PAGE,
    EDGE_DOC_NEXT,
    EDGE_TABLE_HEADER,
    EDGE_TABLE_ROW,
    NODE_METRIC,
    NODE_SPECIES,
    NODE_TEXT_CHUNK,
    GraphStore,
)
from fsqa_heg.retrieval.lexical_table import (
    METRIC_TERMS,
    _is_species_anchor,
    iter_graph_ingest_chunks,
)


def ingest_standards_hetero_graph(
    store: GraphStore,
    indexed: list[dict],
    *,
    lines_per_turn: int = 5,
    table_follow: int = 14,
    use_table_heg: bool = True,
) -> None:
    """Ingest indexed lines as a heterogeneous evidence graph with structural edges."""
    step = 0
    prev_sg: dict[str, Any] | None = None
    prev_page: int | None = None
    last_species_node: dict[str, Any] | None = None
    last_species_page: int | None = None
    seen_metrics: dict[str, str] = {}

    for chunk in iter_graph_ingest_chunks(
        indexed,
        lines_per_turn=lines_per_turn,
        table_follow=table_follow,
        table_aware=use_table_heg,
    ):
        if not chunk:
            continue

        ref_ids = [str(row["ref_id"]) for row in chunk]
        texts = [str(row["text"]) for row in chunk]
        pages = [int(row["page"]) for row in chunk]
        lines = [int(row["line"]) for row in chunk]
        page = pages[0]
        line_start = min(lines)
        line_end = max(lines)

        question = " ".join(ref_ids) + " " + " ".join(texts)
        answer = "\n".join(f"[{rid}] {txt}" for rid, txt in zip(ref_ids, texts))
        step += 1
        sg = store.add_subgraph(
            question=question[:2000],
            answer=answer[:4000],
            observations=[{"ref_ids": ref_ids}],
            step_index=step,
            node_type=NODE_TEXT_CHUNK,
            ref_ids=ref_ids,
            page=page,
            line_start=line_start,
            line_end=line_end,
        )

        if prev_sg is not None:
            store.link_edge(prev_sg["id"], sg["id"], EDGE_DOC_NEXT)
            if prev_page is not None and prev_page == page:
                store.link_edge(prev_sg["id"], sg["id"], EDGE_CO_PAGE, weight=0.35)

        species_rows = [row for row in chunk if _is_species_anchor(str(row.get("text", "")))]
        if use_table_heg and species_rows:
            for row in species_rows:
                anchor = str(row["text"]).strip()
                sp = store.add_evidence_node(
                    node_type=NODE_SPECIES,
                    anchor_text=anchor,
                    ref_ids=[str(row["ref_id"])],
                    page=int(row["page"]),
                    line=int(row["line"]),
                    parent=sg,
                    keywords={anchor},
                )
                store.link_edge(sp["id"], sg["id"], EDGE_CONTAINS)
                store.link_edge(sp["id"], sg["id"], EDGE_TABLE_ROW)
                last_species_node = sp
                last_species_page = int(row["page"])
        elif (
            use_table_heg
            and last_species_node is not None
            and last_species_page is not None
            and page == last_species_page
            and any(not _is_species_anchor(str(row.get("text", ""))) for row in chunk)
        ):
            store.link_edge(last_species_node["id"], sg["id"], EDGE_TABLE_ROW)

        chunk_text = " ".join(texts)
        if use_table_heg:
            for metric in sorted(METRIC_TERMS, key=len, reverse=True):
                if metric not in chunk_text:
                    continue
                metric_key = f"{page}:{metric}"
                if metric_key in seen_metrics:
                    store.link_edge(seen_metrics[metric_key], sg["id"], EDGE_TABLE_HEADER)
                    continue
                mt = store.add_evidence_node(
                    node_type=NODE_METRIC,
                    anchor_text=metric,
                    ref_ids=ref_ids[:1],
                    page=page,
                    line=line_start,
                    parent=sg,
                    keywords={metric},
                )
                store.link_edge(mt["id"], sg["id"], EDGE_TABLE_HEADER)
                store.link_edge(mt["id"], sg["id"], EDGE_CONTAINS)
                seen_metrics[metric_key] = mt["id"]

        prev_sg = sg
        prev_page = page

    store.save()
