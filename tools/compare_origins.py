from __future__ import annotations

from pathlib import Path


def collect_file_names(root: Path) -> set[str]:
    """Return relative file paths (posix style) for every file under root."""
    files = set()
    for file_path in root.rglob("*"):
        if file_path.is_file():
            files.add(file_path.relative_to(root).as_posix())
    return files


def main() -> None:
    base_dir = Path("../data/300class")
    original = base_dir / "origin - 副本"
    target = base_dir / "origin（230节）"

    original_files = collect_file_names(original)
    target_files = collect_file_names(target)

    only_in_original = sorted(original_files - target_files)
    only_in_target = sorted(target_files - original_files)

    print("文件名差异（仅比较路径）：")
    print(f"共比较 {len(original_files)} vs {len(target_files)} 个文件\n")

    print("仅存在于 origin - 副本：")
    if only_in_original:
        for name in only_in_original:
            print(f"  {name}")
    else:
        print("  无")

    print("\n仅存在于 origin（230节）：")
    if only_in_target:
        for name in only_in_target:
            print(f"  {name}")
    else:
        print("  无")


if __name__ == "__main__":
    main()
