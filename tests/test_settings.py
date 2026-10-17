import os

import pytest

from liminal.settings import claude_model, load_env_file, positive_setting


def test_env_file_is_data_and_environment_has_precedence(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('ANTHROPIC_API_KEY="literal-$(touch should-not-exist)"\nANTHROPIC_MODEL=local\n')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "process")
    assert load_env_file(path)
    assert os.environ["ANTHROPIC_API_KEY"] == "literal-$(touch should-not-exist)"
    assert claude_model() == "process"
    monkeypatch.delenv("ANTHROPIC_API_KEY")


def test_env_rejects_unsupported_settings_without_echoing_secret(tmp_path):
    path = tmp_path / ".env"
    path.write_text("UNEXPECTED=secret-value\n")
    with pytest.raises(ValueError, match="supported KEY=value") as error:
        load_env_file(path)
    assert "secret-value" not in str(error.value)


def test_missing_env_and_invalid_budget(tmp_path, monkeypatch):
    assert not load_env_file(tmp_path / "missing")
    monkeypatch.setenv("LIMINAL_MAX_MODEL_CALLS", "0")
    with pytest.raises(ValueError, match="positive"):
        positive_setting("LIMINAL_MAX_MODEL_CALLS", 3)
