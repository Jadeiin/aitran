"""Crowdin API helpers for downloading and uploading translations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from crowdin_api import CrowdinClient
from crowdin_api.api_resources.enums import ExportProjectTranslationFormat


def _translation_format(path: str) -> ExportProjectTranslationFormat:
    """Infer Crowdin translation format from a local path.

    Args:
        path: Local file path.

    Returns:
        Crowdin export format.

    Raises:
        ValueError: If the path extension is unsupported.
    """
    ext = Path(path).suffix.lower()
    if ext not in {".xliff", ".xlf"}:
        raise ValueError("Only .xliff or .xlf files are supported.")
    return ExportProjectTranslationFormat.XLIFF


def _crowdin_base_url_parts(url: str) -> tuple[str, str | None]:
    """Normalize Crowdin base URL and extract its optional scheme.

    Args:
        url: Crowdin base URL override.

    Returns:
        Base URL without scheme, plus the optional URL scheme.

    Raises:
        ValueError: If the URL is empty or uses an unsupported scheme.
    """
    parsed = urlsplit(url.strip().rstrip("/"))
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise ValueError("Crowdin base URL must use http or https.")
    if parsed.scheme or parsed.netloc:
        base_url = f"{parsed.netloc}{parsed.path}"
    else:
        base_url = parsed.path
    base_url = base_url.strip("/")
    if not base_url:
        raise ValueError("Crowdin base URL is required.")
    return f"{base_url}/", parsed.scheme or None


def _items(payload: Any) -> list[dict]:
    """Extract Crowdin SDK list payload items as data dictionaries.

    Args:
        payload: Crowdin SDK list response.

    Returns:
        Plain item dictionaries.
    """
    items = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    return [
        item["data"] if isinstance(item.get("data"), dict) else item
        for item in items
        if isinstance(item, dict)
    ]


def _choice_lines(items: list[dict], *, name_field: str, limit: int) -> str:
    """Format Crowdin objects for a selection error.

    Args:
        items: Crowdin object dictionaries.
        name_field: Preferred display field.
        limit: Maximum number of choices to show.

    Returns:
        Newline-separated choice labels.
    """
    return "\n".join(
        f"  {item.get('id', '?')}: "
        f"{item.get(name_field) or item.get('name') or '<unnamed>'}"
        for item in items[:limit]
    )


def _resolve_project_id(
    client: CrowdinClient,
    *,
    project_id: int | None,
    project: str | None,
) -> int:
    """Resolve a Crowdin project ID from an explicit ID or project name.

    Args:
        client: Crowdin API client.
        project_id: Explicit Crowdin project ID.
        project: Crowdin project name.

    Returns:
        Crowdin project ID.

    Raises:
        ValueError: If the project cannot be resolved unambiguously.
    """
    if project_id is not None:
        return project_id
    if not project:
        raise ValueError("Either Crowdin project ID or project name is required.")

    projects = _items(client.projects.with_fetch_all().list_projects())
    lowered = project.casefold()
    matches = [
        item for item in projects if str(item.get("name", "")).casefold() == lowered
    ]
    if len(matches) == 1:
        return int(matches[0]["id"])

    if matches:
        choices = _choice_lines(matches, name_field="name", limit=20)
        raise ValueError(
            f"Crowdin project name '{project}' is ambiguous. "
            f"Use --project-id with one of:\n{choices}"
        )

    choices = _choice_lines(projects, name_field="name", limit=20)
    raise ValueError(
        f"Crowdin project '{project}' was not found."
        + (f" Available projects:\n{choices}" if choices else "")
    )


def _resolve_file_id(
    client: CrowdinClient,
    *,
    project_id: int,
    file_id: int | None,
) -> int:
    """Resolve or request a Crowdin source file ID.

    Args:
        client: Crowdin API client.
        project_id: Crowdin project ID.
        file_id: Explicit Crowdin file ID.

    Returns:
        Crowdin file ID.

    Raises:
        ValueError: If no file ID was provided.
    """
    if file_id is not None:
        return file_id

    files = _items(client.source_files.list_files(projectId=project_id))
    choices = _choice_lines(files, name_field="path", limit=50)
    raise ValueError(
        "Crowdin file ID is required. Use --file-id with one of:"
        + (f"\n{choices}" if choices else " no files found in this project.")
    )


def _extract_data_field(payload: dict, field: str, context: str) -> Any:
    """Extract a field from a Crowdin API payload.

    Args:
        payload: Crowdin API response dictionary.
        field: Field name to extract.
        context: Label for error messages.

    Returns:
        Field value.

    Raises:
        ValueError: If the field is missing.
    """
    data = payload.get("data")
    if isinstance(data, dict) and field in data:
        return data[field]
    if field in payload:
        return payload[field]
    raise ValueError(f"Missing {context} field '{field}'.")


def _client_url_options(base_url: str | None) -> dict[str, str | None]:
    """Build optional CrowdinClient URL kwargs.

    Args:
        base_url: Optional API base URL override.

    Returns:
        CrowdinClient URL keyword arguments.
    """
    if not base_url:
        return {"base_url": None, "http_protocol": None}
    normalized, scheme = _crowdin_base_url_parts(base_url)
    return {"base_url": normalized, "http_protocol": scheme}


def download_translation(
    *,
    token: str,
    project_id: int | None,
    project: str | None,
    file_id: int | None,
    language: str,
    output_path: str,
    organization: str | None,
    base_url: str | None,
    timeout_seconds: int,
) -> None:
    """Download a translation file from Crowdin.

    Args:
        token: Crowdin API token.
        project_id: Optional Crowdin project ID.
        project: Optional Crowdin project name.
        file_id: Optional Crowdin file ID.
        language: Target language code.
        output_path: Local output file path.
        organization: Crowdin organization (Enterprise only).
        base_url: Optional API base URL override.
        timeout_seconds: Timeout for API operations.

    Raises:
        RequestException: If downloading the build output fails.
    """
    client = CrowdinClient(
        token=token,
        organization=organization,
        project_id=project_id,
        timeout=timeout_seconds,
        **_client_url_options(base_url),
    )
    project_id = _resolve_project_id(
        client,
        project_id=project_id,
        project=project,
    )
    file_id = _resolve_file_id(client, project_id=project_id, file_id=file_id)
    export_payload = client.translations.export_project_translation(
        language,
        projectId=project_id,
        format=_translation_format(output_path),
        fileIds=[file_id],
    )
    url = str(_extract_data_field(export_payload, "url", "export response"))
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise requests.RequestException(
            "Failed to download translation file from Crowdin build URL."
        ) from exc
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)


def upload_translation(
    *,
    token: str,
    project_id: int | None,
    project: str | None,
    file_id: int | None,
    language: str,
    file_path: str,
    organization: str | None,
    base_url: str | None,
    timeout_seconds: int,
) -> None:
    """Upload a translation file to Crowdin.

    Args:
        token: Crowdin API token.
        project_id: Optional Crowdin project ID.
        project: Optional Crowdin project name.
        file_id: Optional Crowdin file ID.
        language: Target language code.
        file_path: Local translation file path.
        organization: Crowdin organization (Enterprise only).
        base_url: Optional API base URL override.
        timeout_seconds: Timeout for API operations.

    """
    _translation_format(file_path)
    client = CrowdinClient(
        token=token,
        organization=organization,
        project_id=project_id,
        timeout=timeout_seconds,
        **_client_url_options(base_url),
    )
    project_id = _resolve_project_id(
        client,
        project_id=project_id,
        project=project,
    )
    file_id = _resolve_file_id(client, project_id=project_id, file_id=file_id)
    with open(file_path, "rb") as handle:
        storage_payload = client.storages.add_storage(handle)
    storage_id = int(_extract_data_field(storage_payload, "id", "storage response"))
    client.translations.upload_translation(
        language,
        storage_id,
        file_id,
        projectId=project_id,
    )
