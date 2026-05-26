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
    """Validate that the file path uses a supported translation extension.

    Args:
        path: File path to validate.

    Raises:
        ValueError: If the file extension is unsupported.
    """
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
    convert: str | None,
) -> None:
    """Download a translation file from Weblate.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        project: Weblate project slug.
        component: Weblate component slug.
        language: Target language code.
        output_path: Local output file path.
        convert: Optional format to convert on the server.

    """
    _ensure_translation_extension(output_path)
    api_url = normalize_weblate_url(url)
    client = Weblate(key=token, url=api_url)
    params = {"format": convert} if convert else None
    content = client.raw_request(
        "GET",
        f"translations/{project}/{component}/{language}/file/",
        params=params,
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
    method: str,
    fuzzy: str | None,
) -> None:
    """Upload a translation file to Weblate.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        project: Weblate project slug.
        component: Weblate component slug.
        language: Target language code.
        file_path: Local translation file path.
        method: Upload method (translate, replace, etc.).
        fuzzy: Optional handling for fuzzy strings.

    """
    _ensure_translation_extension(file_path)
    client = Weblate(key=token, url=normalize_weblate_url(url))
    data = {"method": method}
    if fuzzy:
        data["fuzzy"] = fuzzy
    with open(file_path, "rb") as handle:
        client.request(
            "POST",
            f"translations/{project}/{component}/{language}/upload/",
            files={"file": handle},
            data=data,
        )
