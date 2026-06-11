@echo off
setlocal

set METHODS=direct keyquestion
set SPLITS=train test all
set QUESTION_BANK=
set MAX_ESSAYS=
set HELPER=%~dp0run_essay_tests_single_model.bat
set LOG_DIR=%~dp0..\data\essay\results\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set DEEPSEEK_STATUS=%LOG_DIR%\run_essay_tests_deepseek.status
set GPT_STATUS=%LOG_DIR%\run_essay_tests_gpt.status
set DEEPSEEK_LOG=%LOG_DIR%\run_essay_tests_deepseek.log
set GPT_LOG=%LOG_DIR%\run_essay_tests_gpt.log

if exist "%DEEPSEEK_STATUS%" del "%DEEPSEEK_STATUS%"
if exist "%GPT_STATUS%" del "%GPT_STATUS%"

echo Running full essay comparison tests for deepseek and gpt in parallel...
echo Methods: %METHODS%
echo Splits: %SPLITS%
echo.

start "essay-deepseek" /b cmd /c call "%HELPER%" deepseek deepseek-v3.2 "%METHODS%" "%SPLITS%" "%MAX_ESSAYS%" "%QUESTION_BANK%" "%DEEPSEEK_STATUS%" ^> "%DEEPSEEK_LOG%" 2^>^&1
start "essay-gpt" /b cmd /c call "%HELPER%" gpt gpt-4o-mini "%METHODS%" "%SPLITS%" "%MAX_ESSAYS%" "%QUESTION_BANK%" "%GPT_STATUS%" ^> "%GPT_LOG%" 2^>^&1

echo DeepSeek log: %DEEPSEEK_LOG%
echo GPT log: %GPT_LOG%
echo Waiting for both model runs to finish...

:wait_loop
if not exist "%DEEPSEEK_STATUS%" (
  timeout /t 5 /nobreak >nul
  goto wait_loop
)
if not exist "%GPT_STATUS%" (
  timeout /t 5 /nobreak >nul
  goto wait_loop
)

set /p DEEPSEEK_EXIT=<"%DEEPSEEK_STATUS%"
set /p GPT_EXIT=<"%GPT_STATUS%"

if not "%DEEPSEEK_EXIT%"=="0" (
  echo.
  echo Essay tests failed for deepseek / deepseek-v3.2. Check %DEEPSEEK_LOG%
  exit /b %DEEPSEEK_EXIT%
)

if not "%GPT_EXIT%"=="0" (
  echo.
  echo Essay tests failed for gpt / gpt-4o-mini. Check %GPT_LOG%
  exit /b %GPT_EXIT%
)

echo.
echo Summarizing total test time...
python "%~dp0..\tools\summarize_total_test_time.py"

echo.
echo All essay tests completed.

endlocal
exit /b 0
