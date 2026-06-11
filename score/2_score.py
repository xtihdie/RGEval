from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.live import Live

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pool.runner import ThreadRunner
from test.prompts import scoring_rubric_problem
from evaluation.progress import build_score_progress
from evaluation.question_bank import load_question_dimensions
from evaluation.runtime import add_path_override_argument, add_question_bank_argument, override_config_value, set_question_bank_override

console = Console(force_terminal=True)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("LLM problem-level scoring")
    p.add_argument("--agent", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--num-threads", dest="num_threads", type=int, default=None, help="Override max worker count for this stage.")
    add_path_override_argument(p, "--class-file-path", "class_file_path", "Override classroom/dialogue source directory.")
    add_path_override_argument(p, "--score-file-path", "score_file_path", "Override gold score CSV path.")
    add_path_override_argument(p, "--result-dir", "result_dir", "Override result directory root for this stage.")
    add_question_bank_argument(p)
    return p.parse_args()


# ============================================================
# 配置
# ============================================================

@dataclass
class Config:
    class_file_path: str = str(PROJECT_ROOT / "test/data/300class/origin")
    score_file_path: str = str(PROJECT_ROOT / "test/data/300class/score.csv")
    result_dir: str = str(PROJECT_ROOT / "test/data/300class/results")

    num_threads: int = 6
    tag: str = "2"

    agent_name: str = "deepseek"
    model_name: str = "deepseek3.1"

    output_encoding: str = "utf-8-sig"

    @staticmethod
    def _safe(v: str) -> str:
        return v.replace("/", "-")

    @property
    def result_path(self) -> Path:
        return (
            Path(self.result_dir)
            / self.tag
            / f"{self.tag}_{self._safe(self.agent_name)}_{self._safe(self.model_name)}_scores.csv"
        )


RESULT_COLUMNS = ["课程编号", "维度", "级别", "AI评级", "评语"]

# ============================================================
# 问题维度（保持原样）
# ============================================================

QUESTION_DIMENSIONS: List[str] = []
QUESTION_DIMENSIONS = load_question_dimensions(QUESTION_DIMENSIONS)

# ============================================================
# 工具函数
# ============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"Cannot read {path}")


def read_dialogue_by_code(code: int, root: str) -> Optional[pd.DataFrame]:
    for f in os.listdir(root):
        if f.startswith(str(code)) and f.endswith(".csv"):
            return read_csv_with_auto_encoding(os.path.join(root, f))
    return None


def read_csv_with_auto_encoding(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法识别文件编码: {path}")


def discover_tasks(root: str) -> List[Tuple[int, int]]:
    tasks = []
    for f in os.listdir(root):
        if f.endswith(".csv") and f.split("_")[0].isdigit():
            cid = int(f.split("_")[0])
            for d in range(len(QUESTION_DIMENSIONS)):
                tasks.append((cid, d))
    return sorted(tasks)


def get_origin_level(cid: int, score_df: pd.DataFrame) -> str:
    row = score_df[score_df["课例id"] == cid]
    if row.empty:
        return ""
    v = row.iloc[0].get("级别", "")
    return v if pd.notna(v) else ""


def build_conversation_text(df: pd.DataFrame) -> str:
    return "\n".join(
        f"{r.get('角色','')}：{r.get('内容','')}（{r.get('类别','')}）"
        for _, r in df.iterrows()
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


# ============================================================
# 核心：单问题维度评分（纯函数）
# ============================================================

def grade_single_dimension(
    lesson_id: int,
    dim: int,
    cfg: Config,
    score_df: pd.DataFrame,
    runner: ThreadRunner,
) -> Optional[dict]:

    df = read_dialogue_by_code(lesson_id, cfg.class_file_path)
    if df is None:
        return None

    conv = build_conversation_text(df)
    question = QUESTION_DIMENSIONS[dim]

    messages = [[
        {"role": "system", "content": scoring_rubric_problem["system"]},
        {
            "role": "user",
            "content": scoring_rubric_problem["user"].format(
                class_conv=conv,
                criteria=question,
                score=get_origin_level(lesson_id, score_df),
            ),
        },
    ]]

    try:
        out = runner.run(messages)
        ai_level, comment = parse_llm_response(out[0])
    except Exception:
        return None

    return {
        "课程编号": lesson_id,
        "维度": dim,
        "级别": get_origin_level(lesson_id, score_df),
        "AI评级": ai_level,
        "评语": comment,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()
    set_question_bank_override(args)
    cfg = Config(
        agent_name=args.agent or Config.agent_name,
        model_name=args.model or Config.model_name,
    )
    override_config_value(cfg, args, "num_threads")
    override_config_value(cfg, args, "class_file_path")
    override_config_value(cfg, args, "score_file_path")
    override_config_value(cfg, args, "result_dir")

    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)

    score_df = read_csv_with_auto_encoding(cfg.score_file_path)
    all_tasks = discover_tasks(cfg.class_file_path)

    if cfg.result_path.exists():
        result_df = read_csv_auto(cfg.result_path)
        done = {
            (int(r["课程编号"]), int(r["维度"]))
            for _, r in result_df.iterrows()
        }
    else:
        result_df = pd.DataFrame(columns=RESULT_COLUMNS)
        done = set()

    pending = [t for t in all_tasks if t not in done]
    if not pending:
        return

    runner = ThreadRunner(cfg.agent_name, cfg.model_name, max_workers=cfg.num_threads)
    lock = threading.Lock()

    progress = build_score_progress()

    task_id = progress.add_task(
        "progress",
        total=len(all_tasks),
        completed=len(done),
    )

    with Live(progress, console=console, refresh_per_second=8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_threads) as exe:
            futures = {
                exe.submit(
                    grade_single_dimension,
                    cid,
                    dim,
                    cfg,
                    score_df,
                    runner,
                ): (cid, dim)
                for cid, dim in pending
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
