@echo off
setlocal

set AGENT=qwen
set MODEL=qwen3.5

python "%~dp0..\main.py" ^
  --dataset essay ^
  --stage essay_2 ^
  --agent %AGENT% ^
  --model %MODEL%

endlocal
