@echo off
setlocal

set "AGENT=deepseek"
set "MODEL=deepseek3.1"

python "%~dp0..\main.py" --dataset classroom_dialogue --stage core --agent "%AGENT%" --model "%MODEL%"

