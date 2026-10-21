@echo off
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%"

if "%DEEPSEEK_API_KEY%"=="" (
  echo Please set DEEPSEEK_API_KEY
  exit /b 1
)

if exist "%USERPROFILE%\miniconda3\python.exe" (
  set "PY=%USERPROFILE%\miniconda3\python.exe"
) else (
  set "PY=python"
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
set "SUFFIX=fsqa_ablation"
set "EXTRA=%~2"

if /i "%MODE%"=="all" goto run_all
if /i "%MODE%"=="full" goto run_full
if /i "%MODE%"=="graph_only" goto run_graph
if /i "%MODE%"=="lexical_only" goto run_lex
echo Usage: run_ablations.bat [full^|lexical_only^|graph_only^|all] [--overwrite]
exit /b 1

:run_full
"%PY%" -m fsqa_heg.evaluate --infer-only --skip-classification --context-budget-chars 4000 --budget-strategy lexical_first --scored-truncation --results-suffix %SUFFIX% %EXTRA%
"%PY%" -m fsqa_heg.score --run-tag full --results-suffix %SUFFIX%
goto :eof

:run_lex
"%PY%" -m fsqa_heg.evaluate --lexical-only --infer-only --context-budget-chars 4000 --results-suffix %SUFFIX% %EXTRA%
"%PY%" -m fsqa_heg.score --run-tag lexical_only --results-suffix %SUFFIX%
goto :eof

:run_graph
"%PY%" -m fsqa_heg.evaluate --run-tag graph_only --no-lexical --infer-only --context-budget-chars 4000 --results-suffix %SUFFIX% %EXTRA%
"%PY%" -m fsqa_heg.score --run-tag graph_only --results-suffix %SUFFIX%
goto :eof

:run_all
call "%~f0" full %EXTRA%
call "%~f0" lexical_only %EXTRA%
call "%~f0" graph_only %EXTRA%
"%PY%" "%CD%\fsqa_heg\summarize_ablations.py" --results-suffix %SUFFIX%
goto :eof
