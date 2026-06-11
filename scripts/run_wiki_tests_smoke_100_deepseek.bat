@echo off
setlocal

set AGENT=deepseek
set MODEL=deepseek-v3.2
set METHODS=direct keyquestion
set SPLITS=train test all
set MAX_ARTICLES=100

echo Running smoke wiki comparison tests...
echo Agent: %AGENT%
echo Model: %MODEL%
echo Methods: %METHODS%
echo Splits: %SPLITS%
echo Max articles: %MAX_ARTICLES%
echo.

python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS% --max-articles %MAX_ARTICLES%

if errorlevel 1 (
  echo.
  echo Smoke test failed for %AGENT% / %MODEL%.
  exit /b %errorlevel%
)

echo.
echo Summarizing total test time...
python "%~dp0..\tools\summarize_wiki_test_time.py"

echo.
echo Smoke wiki tests completed.

endlocal
exit /b 0
