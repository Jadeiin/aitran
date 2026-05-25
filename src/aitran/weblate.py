"""Weblate API helpers for downloading and uploading translations."""

from __future__ import annotations

from pathlib import Path

from wlc.client import Weblate

_ALLOWED_EXTENSIONS = {".po", ".pot", ".xliff", ".xlf"}


def normalize_weblate_url(url: str) -> str:
    """Normalize Weblate base URL to the REST API root.

    Returns:
        Normalized API base URL.

    Raises:
        ValueError: If the URL is empty.
    """
    url = url.strip()
    if not url:
        raise ValueError("Weblate URL is required.")
    url = url.rstrip("/")
    if not url.endswith("/api"):
        url = f"{url}/api"
    return f"{url}/"


def _ensure_translation_extension(path: str) -> None:
    ext = Path(path).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError("Only .po, .pot, .xliff, or .xlf files are supported.")


def download_translation(
    *,
    url: str,
    token: str,
    project: str,
    component: str,
    language: str,
    output_path: str,
) -> None:
    """Download a translation file from Weblate."""
    client = Weblate(key=token, url=normalize_weblate_url(url))
    content = client.raw_request(
        "GET",
        f"translations/{project}/{component}/{language}/file/",
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)


def upload_translation(
    *,
    url: str,
    token: str,
    project: str,
    component: str,
    language: str,
    file_path: str,
    replace: bool,
    fuzzy: bool,
) -> None:
    """Upload a translation file to Weblate."""
    _ensure_translation_extension(file_path)
    client = Weblate(key=token, url=normalize_weblate_url(url))
    params = {"replace": str(replace).lower(), "fuzzy": str(fuzzy).lower()}
    with open(file_path, "rb") as handle:
        client.request(
            "POST",
            f"translations/{project}/{component}/{language}/upload/",
            files={"file": handle},
            params=params,
        )
