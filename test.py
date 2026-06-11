# test/test.py
import sys
from pathlib import Path

# ---- 保证能 import 到 llm_pool ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- 只 import ThreadRunner（验证你的目标用法）----
from llm_pool.runner import ThreadRunner


def main():
    # # ===============================
    # # 1) 最懒用法：ds → deepseek → dsv3.1
    # # ===============================
    # print("=== Test 1: ds (default model) ===")
    # runner = ThreadRunner("ds", max_workers=1)

    # messages_list = [
    #     [{"role": "user", "content": "介绍一下你自己"}]
    #     for i in range(1)
    # ]

    # outputs = runner.run(messages_list)
    # print(outputs)

    # # ===============================
    # # 2) 大小写混用测试
    # # ===============================
    # print("=== Test 2: DS / V3.1 (case-insensitive) ===")
    # runner = ThreadRunner("DS", "V3.1", max_workers=2)

    # messages_list = [
    #     [{"role": "user", "content": "hello"}],
    #     [{"role": "user", "content": "world"}],
    # ]

    # outputs = runner.run(messages_list)
    # print(outputs)
    # print()

    # # ===============================
    # # 3) qw → qwen
    # # ===============================
    # print("=== Test 3: qw (qwen) ===")
    # runner = ThreadRunner("qw", max_workers=2)
    
    # messages_list = [
    #     [{"role": "user", "content": "介绍一下你自己"}],
    # ]
    
    # outputs = runner.run(messages_list)
    # print(outputs)

    # ===============================
    # 4) zp → zhipu
    # ===============================
    print("=== Test 4: zp (zhipu) ===")
    runner = ThreadRunner("zp", max_workers=2)
    
    messages_list = [
        [{"role": "user", "content": "介绍一下你自己"}],
    ]
    
    outputs = runner.run(messages_list)
    print(outputs)


if __name__ == "__main__":
    main()
