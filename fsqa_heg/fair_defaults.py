"""林业标准外部基线（A-Mem / MemGPT / Mem0 / dense_rag / HippoRAG / Document GraphRAG）共享公平性超参。"""

from __future__ import annotations

# 与 task_eval/evaluate_standards_qa_mem0.py、dense_rag.py 默认一致
INGEST_LINES_PER_TURN = 5
TABLE_FOLLOW = 14
SEARCH_TOP_K = 15
INJECT_MAX_CHARS = 4000
MAX_OUTPUT_TOKENS = 1024
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_DIMS = 512
