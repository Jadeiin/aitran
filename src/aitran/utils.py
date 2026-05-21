"""File utilities, config discovery, progress bar, language normalization."""

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn


def normalize_lang_code(lang: str) -> str:
    """Normalize language code to lowercase with hyphens (e.g. 'zh_CN' -> 'zh-cn')."""
    return lang.lower().strip().replace(" ", "-").replace("_", "-")


def _git_root() -> str | None:
    """Return the git root directory or None."""
    cwd = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(cwd, ".git")):
            return cwd
        parent = os.path.dirname(cwd)
        if parent == cwd:
            return None
        cwd = parent


def find_config(filename: str) -> str:
    """
    Search for a config file in order:
    1. $CWD/.aitran/
    2. Git root/.aitran/
    3. ~/.aitran/
    Return the first existing path, or the default ~/.aitran/ path.
    """
    home = str(Path.home())
    git_root = _git_root() or os.getcwd()
    candidates = [
        os.path.join(os.getcwd(), ".aitran", filename),
        os.path.join(git_root, ".aitran", filename),
        os.path.join(home, ".aitran", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


def copy_file_if_not_exists(dest: str, src: str, force: bool = False) -> None:
    """Copy src to dest if dest does not exist or is empty (or force=True)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.access(dest, os.F_OK)
        if force or os.stat(dest).st_size == 0:
            shutil.copy2(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def open_file_by_default(filepath: str) -> None:
    """Open a file with the system default application."""
    if platform.system() == "Darwin":
        subprocess.run(["open", filepath])
    elif platform.system() == "Windows":
        os.startfile(filepath)
    else:
        subprocess.run(["xdg-open", filepath])


def open_file_explorer(location: str) -> None:
    """Open the file explorer to the directory containing the given path."""
    dirpath = os.path.dirname(location)
    if platform.system() == "Windows":
        os.startfile(dirpath)
    elif platform.system() == "Darwin":
        subprocess.run(["open", dirpath])
    else:
        subprocess.run(["xdg-open", dirpath])


def create_progress_bar(total: int) -> Progress:
    """Create a rich Progress bar with percentage, bar, and count."""
    return Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
    )


def print_progress(current: int, total: int, extra: str = "") -> None:
    """Simple stderr progress output (used outside of rich context)."""
    percent = current * 100 // max(total, 1)
    bar = "█" * (percent // 5)
    dots = "░" * (20 - percent // 5)
    bar_str = f"{bar}{dots} {percent}% {current}/{total} {extra}"
    os.write(2, f"\r{bar_str}".encode())
    if current >= total:
        os.write(2, b"\n")
