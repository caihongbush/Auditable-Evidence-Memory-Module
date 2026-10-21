# FSQA Dataset (minimal bundle)

| File | Description |
|------|-------------|
| `data/standards_qa.json` | 1,040 QA pairs with page/line gold evidence |
| `data/standards_indexed/*.json` | Line-level parse index per document |

Source PDFs are **not** shipped in this minimal release. Evidence refs use `filename.pdf:P{page}:L{line}`; the index files contain the indexed text at those loci.

## Schema (abbreviated)

```json
{
  "sample_id": "LY1607",
  "source_document": {"file": "LY1607.pdf"},
  "qa": [{
    "question": "...",
    "answer": "...",
    "evidence": ["LY1607.pdf:P1:L6"],
    "category": 1
  }]
}
```

- **category 1**: definitional / factual (n≈1,021)
- **category 2**: procedural (n≈19)

## Metrics

- **Evidence Recall** = set recall over auditable loci
- **Answer F1** = token-level F1 on the short answer
