@echo off
setlocal

set "AGENT=deepseek"
set "MODEL=deepseek3.1"

python "%~dp0..\main.py" --dataset classroom_dialogue --stage 1 --agent "%AGENT%" --model "%MODEL%"
python "%~dp0..\main.py" --dataset classroom_dialogue --stage 1_converge --agent "%AGENT%" --model "%MODEL%"

