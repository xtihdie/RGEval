@echo off
setlocal

set "AGENT=deepseek"
set "MODEL=deepseek3.1"

python "%~dp0..\main.py" --dataset classroom_dialogue --stage 2 --agent "%AGENT%" --model "%MODEL%"
python "%~dp0..\main.py" --dataset classroom_dialogue --stage 2_1_converge --agent "%AGENT%" --model "%MODEL%"
python "%~dp0..\main.py" --dataset classroom_dialogue --stage 2_2_converge --agent "%AGENT%" --model "%MODEL%"

