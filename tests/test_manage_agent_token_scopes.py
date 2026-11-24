import json

import pytest

import scripts.manage_agent_tokens as manage_agent_tokens
from services.agent_auth import AgentTokenRegistry


def test_script_set_scopes_changes_scopes_without_rotating_the_secret(tmp_path, monkeypatch, capsys):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    assert manage_agent_tokens.main(["create", "--name", "review-desk", "--scopes", "read,draft,send",
                                     "--conversations", "72"]) == 0
    created = json.loads(capsys.readouterr().out)
    original_hash = json.loads(path.read_text())["tokens"][0]["hash"]

    assert manage_agent_tokens.main(["set-scopes", "--name", "review-desk", "--scopes", "send,read,draft,note"]) == 0
    assert json.loads(capsys.readouterr().out) == {"name": "review-desk", "scopes": ["draft", "note", "read", "send"]}
    stored = json.loads(path.read_text())["tokens"][0]
    assert stored["hash"] == original_hash
    assert stored["conversations"] == [72]  # the allowlist is untouched
    assert AgentTokenRegistry(path).authenticate(created["token"]).scopes == frozenset({"draft", "note", "read", "send"})


@pytest.mark.parametrize("argv", [["set-scopes", "--name", "nobody", "--scopes", "read"],
                                  ["set-scopes", "--name", "case", "--scopes", "read,admin"],
                                  ["set-scopes", "--name", "case", "--scopes", " , "]])
def test_script_set_scopes_refuses_unknown_names_and_scopes(tmp_path, monkeypatch, capsys, argv):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    assert manage_agent_tokens.main(["create", "--name", "case", "--scopes", "read"]) == 0
    capsys.readouterr()
    original = path.read_bytes()
    with pytest.raises(SystemExit) as refused:
        manage_agent_tokens.main(argv)
    assert refused.value.code == 1
    assert path.read_bytes() == original
