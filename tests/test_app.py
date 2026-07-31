import app
from rag import load_demo_suite


CASE_NAMES = [
    "New employee",
    "Below first threshold",
    "At first threshold",
    "Below second threshold",
    "At second threshold",
]


def test_demo_cli_prints_grounded_passing_report(capsys):
    exit_code = app.main(["--demo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "RuleCheck" in output
    assert "Policy evidence" in output
    assert "Expected" in output
    assert "Actual" in output
    assert "Result" in output
    assert "5/5 cases passed" in output
    for case_name in CASE_NAMES:
        assert case_name in output


def test_live_cli_loads_model_from_project_dotenv(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=fake-test-key\nOPENAI_MODEL=interview-model\n",
        encoding="utf-8",
    )
    captured_models = []

    def capture_generation(evidence, model):
        captured_models.append(model)
        return load_demo_suite(app.DEMO_PATH)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "generate_live_suite", capture_generation)

    exit_code = app.main([])
    capsys.readouterr()

    assert exit_code == 0
    assert captured_models == ["interview-model"]


def test_live_cli_guides_to_demo_when_key_is_missing(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app, "load_dotenv", lambda path: False)

    exit_code = app.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "RuleCheck could not run: OPENAI_API_KEY is not set; "
        "add it to .env or run with --demo\n"
    )


def test_live_cli_returns_runtime_error_without_traceback(monkeypatch, capsys):
    def fail_generation(evidence, model):
        raise RuntimeError("network unavailable")

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setattr(app, "load_dotenv", lambda path: False)
    monkeypatch.setattr(app, "generate_live_suite", fail_generation)

    exit_code = app.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "RuleCheck could not run: network unavailable\n"
