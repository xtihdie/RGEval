from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.console import Console
from rich.live import Live

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pool.runner import ThreadRunner
from test.prompts import converge_scoring_rubric
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_rubric_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI 参数（✅ 与最终模板保持一致）
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM converge scoring runner"
    )

    parser.add_argument(
        "--agent",
        dest="agent_name",
        type=str,
        default=None,
        help="LLM provider name (e.g. deepseek, zhipu, qwen)",
    )

    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=None,
        help="LLM model name",
    )
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=None, help="Override max worker count for this stage.")
    add_path_override_argument(parser, "--class-file-path", "class_file_path", "Override classroom/dialogue source directory.")
    add_path_override_argument(parser, "--score-file-path", "score_file_path", "Override gold score CSV path.")
    add_path_override_argument(parser, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_question_bank_argument(parser)

    return parser.parse_args()


# ============================================================
# 配置
# ============================================================

@dataclass
class Config:
    class_file_path: str = str(PROJECT_ROOT / "test/data/300class/origin")
    score_file_path: str = str(PROJECT_ROOT / "test/data/300class/score.csv")
    result_dir: str = str(PROJECT_ROOT / "test/data/300class/results")

    num_threads: int = 10

    tag: str = "1"
    agent_name: str = "zhipu"
    model_name: str = "glm-4-flash"

    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def result_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_converge.csv"
        )

    @property
    def dimension_scores_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"
        )


RESULT_COLUMNS = ["课程编号", "级别", "AI评级", "评语"]

# ============================================================
# Rubric（保持你当前版本）
# ============================================================

RUBRIC_ITEMS: List[Dict[str, str]] = []
RUBRIC_ITEMS = load_rubric_dimensions(RUBRIC_ITEMS)


def build_rubric_text() -> str:
    return "\n\n".join(
        f"标准({i + 1}) {item['criteria']}:\n\t{item['expound']}"
        for i, item in enumerate(RUBRIC_ITEMS)
    )


RUBRIC_TEXT = build_rubric_text()

# ============================================================
# 工具函数
# ============================================================

def read_csv_with_auto_encoding(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法识别文件编码: {path}")


def discover_all_codes(root: str) -> List[int]:
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(root)
        if f.endswith(".csv") and f.split("_")[0].isdigit()
    )


LEVELS = ("专家级", "中级", "初级")

def parse_llm_response(raw: str):
    # ---------- 1. 标准 <score>...</score> ----------
    m = re.search(
        r"<score[^>]*>\s*\[?\s*(专家级|中级|初级)\s*\]?\s*</score>",
        raw,
        re.I | re.S
    )
    if m:
        level = m.group(1)
    else:
        # ---------- 2. 非标准 <score/>中级<score/> ----------
        m = re.search(
            r"<score[^/>]*/?>\s*\[?\s*(专家级|中级|初级)\s*\]?\s*(?:</?score[^>]*>)?",
            raw,
            re.I | re.S
        )
        if m:
            level = m.group(1)
        else:
            # ---------- 3. 兜底：全文关键词 ----------
            level = None
            for lv in LEVELS:
                if lv in raw:
                    level = lv
                    break

    if not level:
        raise ValueError(f"Cannot parse level from response:\n{raw}")

    # ---------- comment（同样宽松） ----------
    cm = re.search(
        r"<comment[^>]*>\s*\[?(.*?)\]?\s*</comment>",
        raw,
        re.S
    )
    comment = cm.group(1).strip() if cm else ""

    return level, comment
    

def load_dimension_scores(path: Path) -> Dict[int, str]:
    df = read_csv_with_auto_encoding(str(path))
    grouped: Dict[int, str] = {}
    for cid, g in df.groupby("课程编号"):
        lines = []
        for _, r in g.sort_values("维度").iterrows():
            idx = int(r["维度"]) + 1
            lines.append(
                f"标准({idx})：得分：[{r['级别']}]；评语：[{r['评语']}]"
            )
        grouped[int(cid)] = "\n".join(lines)
    return grouped


# ============================================================
# 核心：汇总评分
# ============================================================

def grade_single_lesson(
    lesson_id: int,
    summary: str,
    cfg: Config,
    runner: ThreadRunner,
    score_df: pd.DataFrame,
):
    try:
        messages = [[
            {
                "role": "system",
                "content": converge_scoring_rubric["system"].format(
                    criteria=RUBRIC_TEXT
                ),
            },
            {
                "role": "user",
                "content": converge_scoring_rubric["user"].format(
                    scores_result=summary
                ),
            },
        ]]

        out = runner.run(messages)
        ai_level, comment = parse_llm_response(out[0])

        row = score_df[score_df["课例id"] == lesson_id]
        origin_level = row.iloc[0]["级别"] if not row.empty else ""

        return {
            "课程编号": lesson_id,
            "级别": origin_level,
            "AI评级": ai_level,
            "评语": comment,
        }

    except Exception:
        return None


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)

    cfg = Config(
        agent_name=args.agent_name or Config.agent_name,
        model_name=args.model_name or Config.model_name,
    )
    override_config_value(cfg, args, "num_threads")
    override_config_value(cfg, args, "class_file_path")
    override_config_value(cfg, args, "score_file_path")
    override_config_value(cfg, args, "result_dir")

    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)

    score_df = read_csv_with_auto_encoding(cfg.score_file_path)
    dimension_scores = load_dimension_scores(cfg.dimension_scores_path)
    all_codes = discover_all_codes(cfg.class_file_path)

    if cfg.result_path.exists():
        result_df = read_csv_with_auto_encoding(str(cfg.result_path))
        done = set(result_df["课程编号"].astype(int))
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    pending = [c for c in all_codes if c not in done]

    runner = ThreadRunner(
        cfg.agent_name,
        cfg.model_name,
        max_workers=cfg.num_threads,
    )

    lock = threading.Lock()

    progress = build_score_progress()

    task_id = progress.add_task(
        "progress",
        total=len(all_codes),
        completed=len(done),
    )

    with Live(progress, console=console):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    grade_single_lesson,
                    cid,
                    dimension_scores[cid],
                    cfg,
                    runner,
                    score_df,
                ): cid
                for cid in pending
                if cid in dimension_scores
            }

            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r:
                    with lock:
                        result_df.loc[len(result_df)] = r
                        result_df.to_csv(
                            cfg.result_path,
                            index=False,
                            encoding=cfg.output_encoding,
                        )
                progress.advance(task_id)


if __name__ == "__main__":
    main()
