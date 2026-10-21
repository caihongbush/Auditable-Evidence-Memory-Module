@echo off
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%"

if "%DEEPSEEK_API_KEY%"=="" (
  echo 请先设置 DEEPSEEK_API_KEY
  exit /b 1
)

if exist "%USERPROFILE%\miniconda3\envs\multiagent\python.exe" (
  set "PY=%USERPROFILE%\miniconda3\envs\multiagent\python.exe"
) else if exist "%USERPROFILE%\miniconda3\python.exe" (
  set "PY=%USERPROFILE%\miniconda3\python.exe"
) else (
  set "PY=python"
)

echo === FSQA full system (lexical + HE-G, 4k budget) ===
"%PY%" -m fsqa_heg.evaluate ^
  --infer-only ^
  --skip-classification ^
  --context-budget-chars 4000 ^
  --budget-strategy lexical_first ^
  --scored-truncation ^
  --results-suffix fsqa1k_llm ^
  %*

echo.
echo === Score ===
"%PY%" -m fsqa_heg.score --run-tag full --results-suffix fsqa1k_llm

endlocal
