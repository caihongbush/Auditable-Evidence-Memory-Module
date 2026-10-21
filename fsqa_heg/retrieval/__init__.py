"""Lexical/table primary pathway + HE-G ingest helpers."""

from .heg_ingest import ingest_standards_hetero_graph
from .lexical_table import lexical_retrieve

__all__ = ["ingest_standards_hetero_graph", "lexical_retrieve"]
