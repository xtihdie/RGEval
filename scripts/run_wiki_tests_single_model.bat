@echo off
setlocal

set AGENT=%~1
set MODEL=%~2
set METHODS=%~3
set SPLITS=%~4
set MAX_ARTICLES=%~5
set QUESTION_BANK=%~6
set STATUS_FILE=%~7

echo ========================================
echo Agent: %AGENT%
echo Model: %MODEL%
echo ========================================

if "%QUESTION_BANK%"=="" (
  if "%MAX_ARTICLES%"=="" (
    python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS%
  ) else (
    python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS% --max-articles %MAX_ARTICLES%
  )
) else (
  if "%MAX_ARTICLES%"=="" (
    python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS% --question-bank-path "%QUESTION_BANK%"
  ) else (
    python "%~dp0..\tools\run_wiki_tests.py" --agent %AGENT% --model %MODEL% --methods %METHODS% --splits %SPLITS% --max-articles %MAX_ARTICLES% --question-bank-path "%QUESTION_BANK%"
  )
)

set EXITCODE=%ERRORLEVEL%
> "%STATUS_FILE%" echo %EXITCODE%
exit /b %EXITCODE%
