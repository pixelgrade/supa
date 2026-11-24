import base64
import json
from types import SimpleNamespace

from agents import claude_cli
from services.support_pipeline import SupportPipelineContext, _has_unreviewed_screenshots
from tests.test_support_pipeline import _Investigation, REAL_SHOT


def test_text_mentions_cannot_prove_image_inspection():
    context = SupportPipelineContext(customer_name="Reader", subject="Broken page", message="See screenshot", attachments=[REAL_SHOT])
    assert _has_unreviewed_screenshots(context=context,
        investigation=_Investigation(evidence="The screenshot shows a broken page", reply="Please resend the screenshot"))


def test_claude_receives_pixels_in_the_user_message(monkeypatch, tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2l0kAAAAASUVORK5CYII="))
    captured = {}
    def run(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"result": "answer", "is_error": False}), stderr="")
    monkeypatch.setattr(claude_cli.subprocess, "run", run)
    monkeypatch.setattr(claude_cli, "should_skip_claude", lambda: False)
    result = claude_cli.invoke_claude_result("Inspect these", image_paths=[image])
    content = json.loads(captured["input"])["message"]["content"]
    assert base64.b64decode(content[1]["source"]["data"]) == image.read_bytes()
    assert result["image_input_count"] == 1
    assert "--input-format" in captured["cmd"]


import pytest
from agents import support_agent
from tests.test_deep_pass import _investigation_json


@pytest.mark.parametrize("submitted,held", [(0, True), (1, True), (2, False), (True, True), (-1, True), ("2", True)])
def test_every_real_screenshot_needs_code_recorded_submission(submitted, held):
    context = SupportPipelineContext(customer_name="M", subject="S", message="See images", attachments=[REAL_SHOT, {**REAL_SHOT, "filename": "second.png"}])
    investigation = _Investigation(evidence="I opened every image")
    investigation.image_input_count = submitted
    assert _has_unreviewed_screenshots(context=context, investigation=investigation) is held


def test_model_cannot_claim_submitted_image_count():
    data = json.loads(_investigation_json())
    data["image_input_count"] = 100
    with pytest.raises(ValueError, match="Unexpected fields"):
        support_agent.parse_investigator_result(json.dumps(data))


def test_deep_draft_binds_transport_count_and_forwards_images(monkeypatch, tmp_path):
    captured = {}
    def invoke(**kwargs):
        captured.update(kwargs)
        return {"text": _investigation_json(), "image_input_count": 1}
    monkeypatch.setattr(support_agent, "invoke_claude_result", invoke)
    monkeypatch.setattr(support_agent, "should_skip_claude", lambda: False)
    image = tmp_path / "shot.png"
    image.write_bytes(b"pixels")
    result = support_agent.deep_draft_reply(
        support_agent.SupportInvestigationRequest(customer_name="M", subject="S", message="M"),
        support_agent.DeepPassCritique(previous_draft_html="<p>First</p>", score=7, summary="See images"),
        workdir=tmp_path, image_paths=[image],
    )
    assert captured["image_paths"] == [image]
    assert result.image_input_count == 1


def test_missing_image_never_starts_a_text_only_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_cli, "should_skip_claude", lambda: False)
    monkeypatch.setattr(claude_cli.subprocess, "run", lambda *a, **k: pytest.fail("subprocess started without required image"))
    with pytest.raises(OSError):
        claude_cli.invoke_claude_result("Inspect", image_paths=[tmp_path / "missing.png"])


def test_unsupported_image_data_fails_before_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_cli, "should_skip_claude", lambda: False)
    monkeypatch.setattr(claude_cli.subprocess, "run", lambda *a, **k: pytest.fail("subprocess started with non-image"))
    image = tmp_path / "fake.png"
    image.write_bytes(b"<html>Not an image</html>")
    with pytest.raises(ValueError, match="Unsupported image"):
        claude_cli.invoke_claude_result("Inspect", image_paths=[image])


def test_text_only_invocation_overwrites_forged_transport_count(monkeypatch):
    monkeypatch.setattr(claude_cli, "should_skip_claude", lambda: False)
    monkeypatch.setattr(claude_cli.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps({"result": "answer", "image_input_count": 999}), stderr=""))
    assert claude_cli.invoke_claude_result("Inspect")["image_input_count"] == 0


def test_image_invocation_parses_the_real_cli_result_stream(monkeypatch, tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test pixels")
    captured = {}
    def run(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        stream = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}},
            {"type": "result", "result": "answer", "usage": {"input_tokens": 99}, "is_error": False},
        ]
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(event) for event in stream), stderr="")
    monkeypatch.setattr(claude_cli.subprocess, "run", run)
    monkeypatch.setattr(claude_cli, "should_skip_claude", lambda: False)
    result = claude_cli.invoke_claude_result("Inspect", image_paths=[image])
    assert captured["cmd"][captured["cmd"].index("--output-format")+1] == "stream-json"
    assert "--verbose" in captured["cmd"]
    assert result["text"] == "answer" and result["image_input_count"] == 1
    assert result["usage"] == {"input_tokens": 99}


@pytest.mark.parametrize("submitted,expected_mode", [(1, "draft"), (2, "auto_send")])
def test_partial_download_stays_held_and_complete_submission_can_send(monkeypatch, tmp_path, submitted, expected_mode):
    from services.deep_pass import DeepEvidence
    from services.support_pipeline import run_support_pipeline
    from tests.test_deep_pass import _investigation, _evaluation, ON

    context = SupportPipelineContext(customer_name="M", subject="Broken view", message="See screenshots", attachments=[REAL_SHOT, {**REAL_SHOT, "filename": "second.png"}])
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *a, **k: _investigation(evidence_summary="The screenshot shows a broken view"))
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda *a, **k: _evaluation(9))
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **k: [])
    monkeypatch.setattr("services.support_pipeline.deep_draft_reply", lambda *a, **k: _investigation("<p>Deep draft</p>", image_input_count=submitted))
    def collect(evaluation, workdir):
        paths = []
        for number in range(submitted):
            image = workdir / f"image-{number}.png"
            image.write_bytes(b"downloaded image")
            paths.append(image)
        return DeepEvidence(image_paths=paths)
    result = run_support_pipeline(context=context, max_attempts=1, deep_pass_config=ON, deep_evidence=collect)
    assert result.delivery_mode == expected_mode
    assert ("unreviewed_screenshots" in result.hard_blockers) is (submitted == 1)
    assert result.trace[-1]["image_input_count"] == submitted


def test_unknown_non_signature_inline_image_still_needs_pixels():
    unknown = {"filename": "unknown.png", "inline": True, "is_image": False, "is_signature": False, "is_blank": False}
    context = SupportPipelineContext(customer_name="M", subject="S", message="Broken view", attachments=[unknown])
    assert _has_unreviewed_screenshots(context=context, investigation=_Investigation())


@pytest.mark.asyncio
async def test_collector_downloads_unknown_inline_images_but_not_signatures(monkeypatch, tmp_path):
    from services import processor
    from tests.test_deep_pass import _evaluation
    downloaded = []
    class HelpScout:
        async def get_inline_image_data(self, url, *, max_bytes):
            downloaded.append(url)
            return b"\x89PNG\r\n\x1a\nimage pixels"
    async def expand(**kwargs):
        return {}
    monkeypatch.setattr(processor, "investigate_ticket_for_confidence", expand)
    unknown = {"url": "https://assets.woodland.example/inline/1/unknown.png", "filename": "unknown.png", "inline": True, "is_image": False, "is_signature": False, "is_blank": False}
    signature = {**unknown, "url": unknown["url"].replace("unknown", "logo"), "is_signature": True}
    evidence = await processor.collect_deep_evidence(
        processor=SimpleNamespace(helpscout=HelpScout()),
        ticket={"id": 1, "subject": "S", "message": "M", "attachments": [unknown, signature]},
        evaluation=_evaluation(9), workdir=tmp_path,
    )
    assert downloaded == [unknown["url"]]
    assert len(evidence.image_paths) == 1
    assert evidence.image_paths[0].read_bytes() == b"\x89PNG\r\n\x1a\nimage pixels"


def test_known_signature_and_blank_artifact_never_require_pixels():
    context = SupportPipelineContext(customer_name="M", subject="S", message="M", attachments=[
        {**REAL_SHOT, "is_signature": True},
        {**REAL_SHOT, "is_blank": True},
        {"inline": True, "is_image": False, "is_signature": True, "is_blank": False},
    ])
    assert not _has_unreviewed_screenshots(context=context, investigation=_Investigation())
