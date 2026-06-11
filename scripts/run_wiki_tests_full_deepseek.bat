@echo off
setlocal

set AGENT=deepseek
set MODEL=deepseek-v3.2
set METHODS=direct keyquestion
set SPLITS=test

echo Running full wiki comparison tests...
echo Agent: %AGENT%
echo Model: %MODEL%
echo Methods: %METHODS%
echo Splits: %SPLITS%
echo.

python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS%

if errorlevel 1 (
  echo.
  echo Full test failed for %AGENT% / %MODEL%.
  exit /b %errorlevel%
)

echo.
echo Summarizing total test time...
python "%~dp0..\tools\summarize_wiki_test_time.py"

echo.
echo Full wiki tests completed.

endlocal
exit /b 0
