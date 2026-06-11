from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def build_score_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="bright_magenta"),
        TextColumn("[bold bright_cyan]Scoring in progress[/bold bright_cyan]"),
        BarColumn(bar_width=None, complete_style="bright_cyan", finished_style="bright_green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
