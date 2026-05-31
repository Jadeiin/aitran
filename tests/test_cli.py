"""Tests for the Click CLI wiring."""

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


def test_top_level_app_allows_missing_prompt(monkeypatch):
    calls = []

    def disabled(**_kwargs) -> bool:
        del _kwargs
        return False

    def noop(**_kwargs) -> None:
        del _kwargs

    def fake_run_app(
        prompt,
        *,
        orchestrator_model,
        orchestrator_api_key,
        deps,
        session_id,
        resume,
        auto_approve,
        console,
    ):
        del deps, console
        calls.append((
            prompt,
            orchestrator_model,
            orchestrator_api_key,
            session_id,
            resume,
            auto_approve,
        ))
        return ""

    monkeypatch.setattr(cli, "setup_logfire", disabled)
    monkeypatch.setattr(cli, "flush_logfire", noop)
    monkeypatch.setattr(cli, "setup_mlflow", disabled)
    monkeypatch.setattr(cli, "flush_mlflow", noop)

    import sys
    import types

    fake_module = types.ModuleType("aitran.app")
    fake_module.run_app = fake_run_app
    monkeypatch.setitem(sys.modules, "aitran.app", fake_module)

    result = CliRunner().invoke(cli.app, [], env={})

    assert result.exit_code == 0
    assert len(calls) == 1
    prompt, model, _key, session_id, resume, auto_approve = calls[0]
    assert prompt is None
    assert model == "deepseek:deepseek-v4-pro"
    assert session_id is None
    assert resume is False
    assert auto_approve is False


def test_default_command_launches_app(monkeypatch):
    calls = []

    def disabled(**_kwargs) -> bool:
        del _kwargs
        return False

    def noop(**_kwargs) -> None:
        del _kwargs

    def fake_run_app(
        prompt,
        *,
        orchestrator_model,
        orchestrator_api_key,
        deps,
        session_id,
        resume,
        auto_approve,
        console,
    ):
        del orchestrator_model, orchestrator_api_key, deps, session_id, resume, console
        calls.append((prompt, auto_approve))
        return ""

    monkeypatch.setattr(cli, "setup_logfire", disabled)
    monkeypatch.setattr(cli, "flush_logfire", noop)
    monkeypatch.setattr(cli, "setup_mlflow", disabled)
    monkeypatch.setattr(cli, "flush_mlflow", noop)

    import sys
    import types

    fake_module = types.ModuleType("aitran.app")
    fake_module.run_app = fake_run_app
    monkeypatch.setitem(sys.modules, "aitran.app", fake_module)

    result = CliRunner().invoke(
        cli.app,
        ["--auto-approve", "--prompt", "translate this"],
        env={},
    )

    assert result.exit_code == 0
    assert calls == [("translate this", True)]


def test_top_level_app_reads_app_envvars(monkeypatch):
    calls = []

    def disabled(**_kwargs) -> bool:
        del _kwargs
        return False

    def noop(**_kwargs) -> None:
        del _kwargs

    def fake_run_app(
        prompt,
        *,
        orchestrator_model,
        orchestrator_api_key,
        deps,
        session_id,
        resume,
        auto_approve,
        console,
    ):
        del deps, session_id, resume, console
        calls.append((prompt, orchestrator_model, orchestrator_api_key, auto_approve))
        return ""

    monkeypatch.setattr(cli, "setup_logfire", disabled)
    monkeypatch.setattr(cli, "flush_logfire", noop)
    monkeypatch.setattr(cli, "setup_mlflow", disabled)
    monkeypatch.setattr(cli, "flush_mlflow", noop)

    import sys
    import types

    fake_module = types.ModuleType("aitran.app")
    fake_module.run_app = fake_run_app
    monkeypatch.setitem(sys.modules, "aitran.app", fake_module)

    result = CliRunner().invoke(
        cli.app,
        [],
        env={
            "AITRAN_APP_MODEL": "openai:gpt-5",
            "AITRAN_APP_KEY": "secret",
            "AITRAN_APP_AUTO_APPROVE": "1",
        },
    )

    assert result.exit_code == 0
    assert calls == [(None, "openai:gpt-5", "secret", True)]
