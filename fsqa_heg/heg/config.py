from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_store_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "mswe_graph_memory.json"


@dataclass
class GraphMemoryConfig:
    """Same knobs as mini-swe-agent `GraphMemoryConfig` (standalone dataclass)."""

    enabled: bool = True
    store_path: Path = field(default_factory=_default_store_path)
    retrieve_k: int = 4
    retrieve_radius: float = 12.0
    inject_max_chars: int = 1800
    inject_budget_fraction: float = 0.8
    neighbor_hops: int = 1
    neighbor_k: int = 8
    similarity_threshold_place: float = 0.18
    similarity_threshold_link: float = 0.22
    place_radius: float = 6.0
    activation_decay: float = 0.98
    activation_boost: float = 0.35
    # False = 消融：检索时不使用二维半径过滤与距离惩罚（构图仍保留坐标布局）
    use_spatial_retrieval: bool = True
    # False = 消融：邻域扩散时不走结构边（doc_next / table_row_follow 等），仅走 keyword_similarity
    use_structural_expansion: bool = True
    # False = 消融：HE-G 不使用表格感知构图（行块切分、树种/指标桥接、表格边）
    use_table_heg: bool = True
