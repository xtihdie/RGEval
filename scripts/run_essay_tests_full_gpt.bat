@echo off
setlocal

set AGENT=gpt
set MODEL=gpt-4o-mini
set METHODS=direct keyquestion
set SPLITS=test

echo Running full essay comparison tests...
echo Agent: %AGENT%
echo Model: %MODEL%
echo Methods: %METHODS%
echo Splits: %SPLITS%
echo.

python "%~dp0..\tools\run_essay_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS%
if errorlevel 1 (
  echo.
  echo Full essay test failed for %AGENT% / %MODEL%.
  exit /b %ERRORLEVEL%
)

echo.
echo Summarizing total test time...
python "%~dp0..\tools\summarize_total_test_time.py"
if errorlevel 1 (
  echo.
  echo Failed to summarize essay test time.
  exit /b %ERRORLEVEL%
)

echo.
echo Full essay tests completed for %AGENT% / %MODEL%.

endlocal
exit /b 0
