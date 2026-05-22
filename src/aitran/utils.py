"""File utilities, config discovery, and language normalization."""

import os
import platform
import shutil
import subprocess

import platformdirs


def normalize_lang_code(lang: str) -> str:
    """Normalize language code to lowercase with hyphens (e.g. 'zh_CN' -> 'zh-cn').

    Returns:
        Normalized language code string.
    """
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
    """Search for a config file.

    Lookup order:
    1. $CWD/.aitran/
    2. Git root/.aitran/
    3. User config directory (XDG-compliant, platformdirs)

    Returns:
        The first existing path, or the platformdirs fallback.
    """
    fallback = os.path.join(
        platformdirs.user_config_dir("aitran", ensure_exists=True), filename
    )
    git_root = _git_root() or os.getcwd()
    candidates = [
        os.path.join(os.getcwd(), ".aitran", filename),
        os.path.join(git_root, ".aitran", filename),
        fallback,
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
        subprocess.run(["open", filepath], check=False)
    elif platform.system() == "Windows":
        os.startfile(filepath)
    else:
        subprocess.run(["xdg-open", filepath], check=False)


def open_file_explorer(location: str) -> None:
    """Open the file explorer to the directory containing the given path."""
    dirpath = os.path.dirname(location)
    if platform.system() == "Windows":
        os.startfile(dirpath)
    elif platform.system() == "Darwin":
        subprocess.run(["open", dirpath], check=False)
    else:
        subprocess.run(["xdg-open", dirpath], check=False)
