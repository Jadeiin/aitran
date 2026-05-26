"""Weblate API helpers for downloading and uploading translations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from wlc.client import Translation, Weblate

_DOWNLOAD_FORMATS = {"po", "xliff11", "xliff"}
_OUTPUT_FORMATS = {".po": "po", ".xliff": "xliff", ".xlf": "xliff"}
_ALLOWED_EXTENSIONS = set(_OUTPUT_FORMATS)


def _patch_translation_download() -> None:
    """Patch Weblate Translation.download to support filtered downloads."""

    def _download(
        self: Translation,
        convert: str | None = None,
        q: str | None = None,
    ) -> bytes:
        params = {}
        if convert is not None:
            params["format"] = convert
        if q is not None:
            params["q"] = q

        url = self._get_stored("file_url")
        if params:
            url = f"{url}?{urlencode(params)}"
        return self.weblate.raw_request("get", url)

    Translation.download = _download


_patch_translation_download()


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


def _ensure_list(value: Any) -> list[Any]:
    """Return SDK list-like results as a list.

    Args:
        value: Value returned by an SDK `list()` method.

    Returns:
        A list of objects, wrapping non-iterable objects.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Iterator):
        return list(value)
    return [value]


def list_objects(*, url: str, token: str, object_path: str | None) -> list[Any]:
    """List Weblate objects like `wlc ls`.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        object_path: Optional project/component/translation path.

    Returns:
        Weblate objects returned by the SDK.
    """
    client = Weblate(key=token, url=_normalize_weblate_api_url(url))
    if object_path:
        return _ensure_list(client.get_object(object_path).list())
    return list(client.list_projects())


def get_stats(*, url: str, token: str, object_path: str) -> Any:
    """Return Weblate statistics like `wlc stats`.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        object_path: Project, component, or translation path.

    Returns:
        Statistics returned by the SDK.
    """
    stats = (
        Weblate(key=token, url=_normalize_weblate_api_url(url))
        .get_object(object_path)
        .statistics()
    )
    if isinstance(stats, Iterator):
        return list(stats)
    return stats


def _ensure_translation_extension(path: str) -> None:
    """Validate that the file path uses a supported translation extension.

    Args:
        path: File path to validate.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    ext = Path(path).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError("Only .po, .xliff, or .xlf files are supported.")


def _format_from_output_path(path: str) -> str:
    """Infer Weblate download format from an output path.

    Args:
        path: Output file path.

    Returns:
        Weblate download format matching the output suffix.
    """
    _ensure_translation_extension(path)
    return _OUTPUT_FORMATS[Path(path).suffix.lower()]


def _normalize_download_format(download_format: str | None, output_path: str) -> str:
    """Resolve explicit or inferred Weblate download format.

    Args:
        download_format: Optional Weblate download format.
        output_path: Local output file path.

    Returns:
        Weblate download format to request.

    Raises:
        ValueError: If the requested format is unsupported.
    """
    if download_format is None:
        return _format_from_output_path(output_path)
    if download_format not in _DOWNLOAD_FORMATS:
        raise ValueError("Only po, xliff11, or xliff download formats are supported.")
    return download_format


def download_translation(
    *,
    url: str,
    token: str,
    object_path: str,
    output_path: str,
    download_format: str | None,
    untranslated_only: bool,
) -> None:
    """Download a translation file from Weblate.

    Args:
        url: Weblate base URL.
        token: Weblate API token.
        object_path: Weblate translation object path (<project>/<component>/<language>).
        output_path: Local output file path.
        download_format: Optional format to download from the server.
        untranslated_only: Whether to download only untranslated strings.

    Raises:
        TypeError: If object path does not target a translation resource.

    """
    resolved_format = _normalize_download_format(download_format, output_path)
    client = Weblate(key=token, url=_normalize_weblate_api_url(url))
    obj = client.get_object(object_path)
    if not isinstance(obj, Translation):
        raise TypeError(
            "Weblate object path must point to a translation resource "
            "(<project>/<component>/<language>)."
        )
    content = obj.download(
        resolved_format,
        q="is:untranslated" if untranslated_only else None,
    )
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
