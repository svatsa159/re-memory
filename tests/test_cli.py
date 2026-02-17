"""CLI smoke tests."""

from typer.testing import CliRunner

from re_memory.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_init(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Initialized" in result.output or "initialized" in result.output.lower()


def test_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Init first
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "brain-anatomical" in result.output.lower() or "Usage" in result.output or "usage" in result.output.lower()
