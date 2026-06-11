@echo off
setlocal

set AGENT=gpt
set MODEL=gpt-4o-mini
set METHODS=direct keyquestion
set SPLITS=train test all
set MAX_ESSAYS=300

echo Running smoke essay comparison tests...
echo Agent: %AGENT%
echo Model: %MODEL%
echo Methods: %METHODS%
echo Splits: %SPLITS%
echo Max essays: %MAX_ESSAYS%
echo.

python "%~dp0..\tools\run_essay_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS% --max-essays %MAX_ESSAYS%

if errorlevel 1 (
  echo.
  echo Smoke test failed for %AGENT% / %MODEL%.
  exit /b %errorlevel%
)

echo.
echo Summarizing total test time...
python "%~dp0..\tools\summarize_total_test_time.py"

echo.
echo Smoke essay tests completed.

endlocal
exit /b 0
