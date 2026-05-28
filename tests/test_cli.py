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
    )

    assert result.exit_code == 0
    assert calls[0] == ("setup_logfire", False, False)
    assert calls[1] == ("setup_mlflow", True, "http://localhost:5000", "aitran-test")
    assert calls[2] == ("translate_po", str(po_file))
    assert calls[3] == ("flush_logfire", False)
    assert calls[4] == ("flush_mlflow", True)
