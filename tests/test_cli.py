"""Tests for the Click CLI wiring."""

import pytest
from click.testing import CliRunner

from aitran import cli
from aitran.agent_work import BatchProgress


def test_userdict_opens_the_dictionary_used_for_normalized_language(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *_args, **_kwargs: str(tmp_path)
    )
    config_dir = tmp_path / ".aitran"
    config_dir.mkdir()
    dictionary_path = config_dir / "dictionary-zh-cn.json"
    dictionary_path.write_text('{"hello": "你好"}', encoding="utf-8")
    opened = []
    monkeypatch.setattr(cli, "open_file_by_default", opened.append)

    result = CliRunner().invoke(cli.app, ["userdict", "-l", "zh_CN"], env={})

    assert result.exit_code == 0
    assert opened == [str(dictionary_path)]


def test_headless_progress_is_written_to_stderr(capsys):
    with cli._progress_reporter() as report:
        report(
            "messages.po",
            BatchProgress(accepted=25, total=100, remaining=75, complete=False),
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "messages.po" in captured.err
    assert "25%" in captured.err
    assert "25/100" in captured.err


def test_headless_progress_keeps_one_rich_task_per_file(capsys):
    with cli._progress_reporter() as report:
        report(
            "first.po",
            BatchProgress(accepted=1, total=2, remaining=1, complete=False),
        )
        report(
            "second.po",
            BatchProgress(accepted=1, total=3, remaining=2, complete=False),
        )
        report(
            "first.po",
            BatchProgress(accepted=2, total=2, remaining=0, complete=True),
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("first.po") == 1
    assert captured.err.count("second.po") == 1
    assert "2/2" in captured.err
    assert "1/3" in captured.err


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

    def fake_translate_file(**kwargs):
        calls.append(("translate_file", kwargs["input_path"]))
        return {"output_path": kwargs["output_path"]}

    monkeypatch.setattr(cli, "setup_logfire", fake_setup_logfire)
    monkeypatch.setattr(cli, "flush_logfire", fake_flush_logfire)
    monkeypatch.setattr(cli, "setup_mlflow", fake_setup_mlflow)
    monkeypatch.setattr(cli, "flush_mlflow", fake_flush_mlflow)
    monkeypatch.setattr(cli, "translate_file", fake_translate_file)

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
    assert calls[2] == ("translate_file", str(po_file))
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

    def fake_translate_file(**kwargs):
        calls.append(("translate_file", kwargs["input_path"]))
        return {"output_path": kwargs["output_path"]}

    monkeypatch.setattr(cli, "setup_logfire", fake_setup_logfire)
    monkeypatch.setattr(cli, "flush_logfire", fake_flush_logfire)
    monkeypatch.setattr(cli, "setup_mlflow", fake_setup_mlflow)
    monkeypatch.setattr(cli, "flush_mlflow", fake_flush_mlflow)
    monkeypatch.setattr(cli, "translate_file", fake_translate_file)

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
    assert calls[2] == ("translate_file", str(po_file))
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
            "AITRAN_TRANSLATOR_MODEL": "anthropic:translator",
            "AITRAN_TRANSLATOR_KEY": "translator-key",
            "AITRAN_REVIEWER_MODEL": "openai:reviewer",
            "AITRAN_REVIEWER_KEY": "reviewer-key",
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
    assert call["deps"].translate_model == "anthropic:translator"
    assert call["deps"].translate_api_key == "translator-key"
    assert call["deps"].review_model == "openai:reviewer"
    assert call["deps"].review_api_key == "reviewer-key"
