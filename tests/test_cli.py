"""Tests for the Click CLI wiring."""

import pytest
from click.testing import CliRunner

from aitran import cli


def test_translate_command_enables_logfire(monkeypatch, tmp_path):
    calls = []
    po_file = tmp_path / "messages.po"
    po_file.write_text("", encoding="utf-8")

    def fake_setup_logfire(*, enabled, capture_http):
        calls.append(("setup_logfire", enabled, capture_http))
        return enabled

    def fake_flush_logfire(*, enabled):
        calls.append(("flush_logfire", enabled))

    def fake_setup_mlflow(*, enabled, tracking_uri=None, experiment=None):
        calls.append(("setup_mlflow", enabled, tracking_uri, experiment))
        return enabled

    def fake_flush_mlflow(*, enabled):
        calls.append(("flush_mlflow", enabled))

    def fake_translate_po(**kwargs):
        calls.append(("translate_po", kwargs["po_path"]))

    monkeypatch.setattr(cli, "setup_logfire", fake_setup_logfire)
    monkeypatch.setattr(cli, "flush_logfire", fake_flush_logfire)
    monkeypatch.setattr(cli, "setup_mlflow", fake_setup_mlflow)
    monkeypatch.setattr(cli, "flush_mlflow", fake_flush_mlflow)
    monkeypatch.setattr(cli, "translate_po", fake_translate_po)

    result = CliRunner().invoke(
        cli.app,
        [
            "translate",
            "--po",
            str(po_file),
            "-l",
            "zh_CN",
            "--logfire",
            "--logfire-capture-http",
        ],
        env={},
    )

    assert result.exit_code == 0
    assert calls[0] == ("setup_logfire", True, True)
    assert calls[1] == ("setup_mlflow", False, None, None)
    assert calls[2] == ("translate_po", str(po_file))
    assert calls[3] == ("flush_logfire", True)
    assert calls[4] == ("flush_mlflow", False)


def test_translate_command_enables_mlflow(monkeypatch, tmp_path):
    calls = []
    po_file = tmp_path / "messages.po"
    po_file.write_text("", encoding="utf-8")

    def fake_setup_logfire(*, enabled, capture_http):
        calls.append(("setup_logfire", enabled, capture_http))
        return enabled

    def fake_flush_logfire(*, enabled):
        calls.append(("flush_logfire", enabled))

    def fake_setup_mlflow(*, enabled, tracking_uri=None, experiment=None):
        calls.append(("setup_mlflow", enabled, tracking_uri, experiment))
        return enabled

    def fake_flush_mlflow(*, enabled):
        calls.append(("flush_mlflow", enabled))

    def fake_translate_po(**kwargs):
        calls.append(("translate_po", kwargs["po_path"]))

    monkeypatch.setattr(cli, "setup_logfire", fake_setup_logfire)
    monkeypatch.setattr(cli, "flush_logfire", fake_flush_logfire)
    monkeypatch.setattr(cli, "setup_mlflow", fake_setup_mlflow)
    monkeypatch.setattr(cli, "flush_mlflow", fake_flush_mlflow)
    monkeypatch.setattr(cli, "translate_po", fake_translate_po)

    result = CliRunner().invoke(
        cli.app,
        [
            "translate",
            "--po",
            str(po_file),
            "-l",
            "zh_CN",
            "--mlflow",
            "--mlflow-tracking-uri",
            "http://localhost:5000",
            "--mlflow-experiment",
            "aitran-test",
        ],
        env={},
    )

    assert result.exit_code == 0
    assert calls[0][0] == "setup_logfire"
    assert calls[0][2] is False
    assert calls[1] == ("setup_mlflow", True, "http://localhost:5000", "aitran-test")
    assert calls[2] == ("translate_po", str(po_file))
    assert calls[3][0] == "flush_logfire"
    assert calls[4] == ("flush_mlflow", True)


def _install_fake_web_module(monkeypatch, calls):
    import sys
    import types

    def fake_run_web_app(**kwargs):
        calls.append(kwargs)

    fake_module = types.ModuleType("aitran.app")
    fake_module.run_web_app = fake_run_web_app
    monkeypatch.setitem(sys.modules, "aitran.app", fake_module)


def _disable_observability(monkeypatch):
    monkeypatch.setattr(cli, "setup_logfire", lambda **_kwargs: False)
    monkeypatch.setattr(cli, "flush_logfire", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "setup_mlflow", lambda **_kwargs: False)
    monkeypatch.setattr(cli, "flush_mlflow", lambda **_kwargs: None)


def test_default_command_launches_webui(monkeypatch):
    calls = []
    _disable_observability(monkeypatch)
    _install_fake_web_module(monkeypatch, calls)

    result = CliRunner().invoke(cli.app, [], env={})

    assert result.exit_code == 0
    assert calls[0]["orchestrator_model"] == "deepseek:deepseek-v4-pro"
    assert calls[0]["orchestrator_temperature"] == pytest.approx(0.5)
    assert calls[0]["web_host"] == "127.0.0.1"
    assert calls[0]["web_port"] == 7932


def test_top_level_webui_options(monkeypatch):
    calls = []
    _disable_observability(monkeypatch)
    _install_fake_web_module(monkeypatch, calls)

    result = CliRunner().invoke(
        cli.app, ["--web-host", "0.0.0.0", "--web-port", "9000"], env={}
    )

    assert result.exit_code == 0
    assert calls[0]["web_host"] == "0.0.0.0"
    assert calls[0]["web_port"] == 9000


def test_top_level_app_reads_app_envvars(monkeypatch):
    calls = []
    _disable_observability(monkeypatch)
    _install_fake_web_module(monkeypatch, calls)

    result = CliRunner().invoke(
        cli.app,
        [],
        env={
            "AITRAN_APP_MODEL": "openai:gpt-5",
            "AITRAN_APP_KEY": "secret",
            "AITRAN_APP_HOST": "https://example.com/api",
            "AITRAN_APP_TMP": "0.4",
            "AITRAN_WEB_HOST": "0.0.0.0",
            "AITRAN_WEB_PORT": "8123",
        },
    )

    assert result.exit_code == 0
    call = calls[0]
    assert call["orchestrator_model"] == "openai:gpt-5"
    assert call["orchestrator_api_key"] == "secret"
    assert call["orchestrator_api_host"] == "https://example.com/api"
    assert call["orchestrator_temperature"] == pytest.approx(0.4)
    assert call["web_host"] == "0.0.0.0"
    assert call["web_port"] == 8123
