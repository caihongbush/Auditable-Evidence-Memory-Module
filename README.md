# SEN-AEG: Structured Evidence Navigation on FSQA

Minimal release: **main method**, **FSQA dataset** (annotations + line index, no source PDFs), and **evaluation scripts**.

## Layout

```
SEN-AGE/
├── fsqa_heg/                 # Main method (lexical-first + structure-guided CPG)
├── data/
│   ├── standards_qa.json     # 1,040 QA pairs + gold page/line evidence
│   └── standards_indexed/    # Parsed line index (required for retrieval)
├── scripts/
│   ├── run_ablations.bat     # full / lexical_only / graph_only
│   └── ABLATIONS.md
├── requirements.txt
├── DATASET.md
└── LICENSE.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # set DEEPSEEK_API_KEY
export PYTHONPATH=$PWD        # Windows: set PYTHONPATH=%CD%
```

## Evaluate

**Full system (1,040 questions):**

```bash
fsqa_heg/run_full_fsqa.bat
# or:
python -m fsqa_heg.evaluate --infer-only --skip-classification \
  --context-budget-chars 4000 --budget-strategy lexical_first \
  --scored-truncation --results-suffix fsqa1k_llm
python -m fsqa_heg.score --run-tag full --results-suffix fsqa1k_llm
```

**Pathway ablations:**

```bash
scripts/run_ablations.bat all
```

Results go to `outputs/` (auto-created).

## Dataset note

Source forestry-standard PDFs are **not** included. You need `data/standards_indexed/` (bundled) to run evaluation. See `DATASET.md` for the JSON schema.

## License

CC BY-NC 4.0 — see `LICENSE.txt`.
