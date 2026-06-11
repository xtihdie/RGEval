from __future__ import annotations

import re


DEFAULT_LEVELS: tuple[str, ...] = ("专家级", "中级", "初级")


def parse_level_response(raw: str, levels: tuple[str, ...] = DEFAULT_LEVELS) -> tuple[str, str]:
    level_pattern = "|".join(re.escape(level) for level in levels)

    match = re.search(
        rf"<score[^>]*>\s*\[?\s*({level_pattern})\s*\]?\s*</score>",
        raw,
        re.I | re.S,
    )
    if not match:
        match = re.search(
            rf"<score[^/>]*/?>\s*\[?\s*({level_pattern})\s*\]?\s*(?:</?score[^>]*>)?",
            raw,
            re.I | re.S,
        )

    if match:
        level = match.group(1)
    else:
        level = next((candidate for candidate in levels if candidate in raw), "")

    if not level:
        raise ValueError(f"Cannot parse level from response:\n{raw}")

    comment_match = re.search(r"<comment[^>]*>\s*\[?(.*?)\]?\s*</comment>", raw, re.S)
    comment = comment_match.group(1).strip() if comment_match else ""
    return level, comment


def parse_numeric_score_tags(raw: str) -> list[int]:
    return [int(value) for value in re.findall(r"<score>\s*\[?\s*(\d{1,3})", raw or "")]
