"""Crowdin API helpers for downloading and uploading translations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from crowdin_api import CrowdinClient

_ALLOWED_EXTENSIONS = {".po", ".pot", ".xliff", ".xlf"}


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


def _extract_data_field(payload: dict, field: str, context: str) -> Any:
    """Extract a field from a Crowdin API payload.

    Args:
        payload: Crowdin API response dictionary.
        field: Field name to read.
        context: Context label for error messages.

    Returns:
        Extracted field value.

    Raises:
        ValueError: If the field is missing.
    """
    data = payload.get("data")
    if isinstance(data, dict) and field in data:
        return data[field]
    if field in payload:
        return payload[field]
    raise ValueError(f"Missing {context} field '{field}'.")


def _wait_for_build(
    client: CrowdinClient,
    *,
    build_id: int,
    project_id: int,
    timeout_seconds: int,
    poll_interval: int,
) -> None:
    """Poll Crowdin build status until completion or failure.

    Args:
        client: Crowdin API client.
        build_id: Build identifier returned by Crowdin.
        project_id: Crowdin project ID.
        timeout_seconds: Maximum time to wait.
        poll_interval: Sleep interval between status checks.

    Raises:
        TimeoutError: If the build does not finish in time.
        ValueError: If the build ends in failed or canceled state.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        status_payload = client.translations.check_project_build_status(
            build_id, projectId=project_id
        )
        status = _extract_data_field(status_payload, "status", "build status")
        if status == "finished":
            return
        if status in {"failed", "canceled"}:
            raise ValueError(f"Crowdin build {build_id} ended with status '{status}'.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Crowdin build to finish.")
        time.sleep(poll_interval)


def download_translation(
    *,
    token: str,
    project_id: int,
    file_id: int,
    language: str,
    output_path: str,
    organization: str | None,
    base_url: str | None,
    timeout_seconds: int,
    poll_interval: int,
) -> None:
    """Download a translation file from Crowdin.

    Args:
        token: Crowdin API token.
        project_id: Crowdin project ID.
        file_id: Crowdin file ID.
        language: Target language code.
        output_path: Local output file path.
        organization: Crowdin organization (Enterprise only).
        base_url: Optional API base URL override.
        timeout_seconds: Timeout for API operations.
        poll_interval: Polling interval for build completion.

    Raises:
        RequestException: If downloading the build output fails.
    """
    client = CrowdinClient(
        token=token,
        organization=organization,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout_seconds,
    )
    build_payload = client.translations.build_project_file_translation(
        file_id,
        language,
        projectId=project_id,
    )
    build_id = int(_extract_data_field(build_payload, "id", "build response"))
    _wait_for_build(
        client,
        build_id=build_id,
        project_id=project_id,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
    )
    download_payload = client.translations.download_project_translations(
        build_id, projectId=project_id
    )
    url = _extract_data_field(download_payload, "url", "download response")
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
    project_id: int,
    file_id: int,
    language: str,
    file_path: str,
    organization: str | None,
    base_url: str | None,
    timeout_seconds: int,
) -> None:
    """Upload a translation file to Crowdin.

    Args:
        token: Crowdin API token.
        project_id: Crowdin project ID.
        file_id: Crowdin file ID.
        language: Target language code.
        file_path: Local translation file path.
        organization: Crowdin organization (Enterprise only).
        base_url: Optional API base URL override.
        timeout_seconds: Timeout for API operations.

    """
    _ensure_translation_extension(file_path)
    client = CrowdinClient(
        token=token,
        organization=organization,
        base_url=base_url,
        project_id=project_id,
        timeout=timeout_seconds,
    )
    with open(file_path, "rb") as handle:
        storage_payload = client.storages.add_storage(handle)
    storage_id = int(_extract_data_field(storage_payload, "id", "storage response"))
    client.translations.upload_translation(
        language,
        storage_id,
        file_id,
        projectId=project_id,
    )
