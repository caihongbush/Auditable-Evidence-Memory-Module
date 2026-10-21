from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import GraphMemoryConfig
from .content_string import get_content_string
from .graph_store import GraphStore

MEMORY_HEADER = "## Graph memory (heterogeneous evidence + 2D proximity)"


class GraphMemoryRuntime:
    """Inject / strip graph-memory snippets around an LM call; persist turns via `GraphStore`."""

    def __init__(self, config: GraphMemoryConfig | None = None):
        self.config = config or GraphMemoryConfig()
        self._store: GraphStore | None = None

    @property
    def store(self) -> GraphStore:
        if self._store is None:
            mem = self.config
            self._store = GraphStore(
                Path(mem.store_path),
                similarity_threshold_place=float(mem.similarity_threshold_place),
                similarity_threshold_link=float(mem.similarity_threshold_link),
                place_radius=float(mem.place_radius),
                activation_decay=float(mem.activation_decay),
                activation_beta=float(mem.activation_boost),
                use_spatial_retrieval=bool(mem.use_spatial_retrieval),
                use_structural_expansion=bool(mem.use_structural_expansion),
                use_table_heg=bool(mem.use_table_heg),
            )
        return self._store

    def build_memory_snippet(
        self, user_query: str, *, query_hints: dict[str, Any] | None = None
    ) -> str | None:
        mem = self.config
        if not mem.enabled:
            return None
        seeds, _bridges, neighbors = self.store.query_retrieval(
            user_query,
            k=int(mem.retrieve_k),
            radius=float(mem.retrieve_radius),
            query_hints=query_hints,
            hops=int(mem.neighbor_hops),
            max_neighbors=int(mem.neighbor_k),
        )
        if not seeds and not neighbors:
            return None
        text = self.store.to_prompt_snippet(
            list(seeds) + list(neighbors),
            max_chars=int(mem.inject_max_chars),
            budget_fraction=float(mem.inject_budget_fraction),
        )
        return text if text.strip() else None

    def append_memory_user_message(self, messages: list[dict[str, Any]], snippet: str) -> int:
        messages.append({"role": "user", "content": snippet})
        return len(messages) - 1

    def remove_memory_message(self, messages: list[dict[str, Any]], index: int | None) -> None:
        if index is None or not (0 <= index < len(messages)):
            return
        msg = messages[index]
        if get_content_string(msg).startswith(MEMORY_HEADER):
            messages.pop(index)

    def record_turn(
        self,
        *,
        question: str,
        answer: str,
        actions: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
        step_index: int | None = None,
        node_type: str | None = None,
        ref_ids: list[str] | None = None,
        page: int | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        kwargs: dict[str, Any] = {}
        if node_type is not None:
            kwargs["node_type"] = node_type
        if ref_ids is not None:
            kwargs["ref_ids"] = ref_ids
        if page is not None:
            kwargs["page"] = page
        if line_start is not None:
            kwargs["line_start"] = line_start
        if line_end is not None:
            kwargs["line_end"] = line_end
        self.store.add_subgraph(
            question=question,
            answer=answer,
            actions=actions,
            observations=observations,
            step_index=step_index,
            **kwargs,
        )

    def latest_user_text(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if (msg.get("role") or msg.get("type")) == "user":
                return get_content_string(msg)
        return ""
