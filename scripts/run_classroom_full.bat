@echo off
setlocal

set "AGENT=deepseek"
set "MODEL=deepseek3.1"
set "AGENT_A=deepseek"
set "MODEL_A=deepseek3.1"
set "AGENT_B=zhipu"
set "MODEL_B=glm-4-flash"

python "%~dp0..\main.py" --dataset classroom_dialogue --stage full --agent "%AGENT%" --model "%MODEL%" --agent-a "%AGENT_A%" --model-a "%MODEL_A%" --agent-b "%AGENT_B%" --model-b "%MODEL_B%"
