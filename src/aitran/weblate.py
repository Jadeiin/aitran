"""Weblate API helpers for downloading and uploading translations."""

from __future__ import annotations

from pathlib import Path

from wlc.client import Translation, Weblate

_ALLOWED_EXTENSIONS = {".po", ".xliff"}


def _normalize_weblate_api_url(url: str) -> str:
    """Normalize Weblate base URL to API root.

    Args:
        url: Weblate base URL.

    Returns:
        Normalized API URL ending with `/api/`.

    Raises:
        ValueError: If URL is empty.
    """
    api_url = url.strip().rstrip("/")
    if not api_url:
        raise ValueError("Weblate URL is required.")
    if not api_url.endswith("/api"):
        api_url = f"{api_url}/api"
    return f"{api_url}/"


def _ensure_translation_extension(path: str) -> None:
    """Validate that the file path uses a supported translation extension.

    Args:
        path: File path to validate.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    ext = Path(path).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError("Only .po or .xliff files are supported.")


def download_translation(
    *,
    url: str,
    token: str,
    object_path: str,
    output_path: str,
    convert: str | None,
) -> None:
    """Download a translation file from Weblate.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        object_path: Weblate translation object path (<project>/<component>/<language>).
        output_path: Local output file path.
        convert: Optional format to convert on the server.

    Raises:
        TypeError: If object path does not target a translation resource.

    """
    _ensure_translation_extension(output_path)
    client = Weblate(key=token, url=_normalize_weblate_api_url(url))
    obj = client.get_object(object_path)
    if not isinstance(obj, Translation):
        raise TypeError(
            "Weblate object path must point to a translation resource "
            "(<project>/<component>/<language>)."
        )
    content = obj.download(convert)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)


def upload_translation(
    *,
    url: str,
    token: str,
    object_path: str,
    file_path: str,
    method: str,
    fuzzy: str | None,
) -> None:
    """Upload a translation file to Weblate.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        object_path: Weblate translation object path (<project>/<component>/<language>).
        file_path: Local translation file path.
        method: Upload method (translate, replace, etc.).
        fuzzy: Optional handling for fuzzy strings.

    Raises:
        TypeError: If object path does not target a translation resource.

    """
    _ensure_translation_extension(file_path)
    client = Weblate(key=token, url=_normalize_weblate_api_url(url))
    obj = client.get_object(object_path)
    if not isinstance(obj, Translation):
        raise TypeError(
            "Weblate object path must point to a translation resource "
            "(<project>/<component>/<language>)."
        )
    data: dict[str, str] = {"method": method}
    if fuzzy:
        data["fuzzy"] = fuzzy
    with open(file_path, "rb") as handle:
        obj.upload(handle, **data)
