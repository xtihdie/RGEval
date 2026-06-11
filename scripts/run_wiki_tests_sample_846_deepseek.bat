@echo off
setlocal

set AGENT=deepseek
set MODEL=deepseek-v3.2
set METHODS=direct keyquestion
set SPLITS=test
set MAX_ARTICLES=846

echo Running sampled wiki comparison tests...
echo Agent: %AGENT%
echo Model: %MODEL%
echo Methods: %METHODS%
echo Splits: %SPLITS%
echo Max articles: %MAX_ARTICLES%
echo.

python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS% --max-articles %MAX_ARTICLES%
if errorlevel 1 (
  echo.
  echo Sampled wiki test failed for %AGENT% / %MODEL%.
  exit /b %ERRORLEVEL%
)

echo.
echo Summarizing total test time...
python "%~dp0..\tools\summarize_wiki_test_time.py"
if errorlevel 1 (
  echo.
  echo Failed to summarize wiki test time.
  exit /b %ERRORLEVEL%
)

echo.
echo Sampled wiki tests completed for %AGENT% / %MODEL%.

endlocal
exit /b 0
