@echo off
setlocal

set "AGENT_A=deepseek"
set "MODEL_A=deepseek3.1"
set "AGENT_B=zhipu"
set "MODEL_B=glm-4-flash"

python "%~dp0..\main.py" --dataset classroom_dialogue --stage 4_1 --agent-a "%AGENT_A%" --model-a "%MODEL_A%" --agent-b "%AGENT_B%" --model-b "%MODEL_B%"

