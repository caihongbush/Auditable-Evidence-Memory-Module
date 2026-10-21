# FSQA pathway ablations (minimal)

Uses `fsqa_heg` only (no external baselines).

```bat
set DEEPSEEK_API_KEY=sk-...
set PYTHONPATH=%CD%
scripts\run_ablations.bat all
```

| Command | Configuration |
|---------|----------------|
| `run_ablations.bat full` | Lexical + structure-guided (full system) |
| `run_ablations.bat lexical_only` | Primary pathway only |
| `run_ablations.bat graph_only` | Structure-guided CPG only |
| `run_ablations.bat all` | All three + summary table |

Add `--overwrite` as second argument to re-run inference.
