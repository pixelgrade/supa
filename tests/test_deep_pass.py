import asyncio
from pathlib import Path

import pytest

from agents import support_agent
from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import (
    DeepPassConfig,
    DeepPassCritique,
    SupportInvestigationRequest,
    SupportInvestigationResult,
    fallback_investigator_result,
)
from services import processor as processor_module
from services.deep_pass import DeepEvidence, cited_source_excerpts, named_theme_slugs, deep_actions, other_customer_emails
from services.support_pipeline import SupportPipelineContext, run_support_pipeline

ON = DeepPassConfig(enabled=True, effort="high", timeout=60)
OFF = DeepPassConfig(enabled=False, effort="high", timeout=60)


def _investigation(reply="<p>First draft</p>", **kwargs) -> SupportInvestigationResult:
    base = dict(
        reply_html=reply,
        confidence_score=7,
        decision_summary="Investigation summary",
        evidence_summary="Docs evidence",
        investigation_steps=["Checked docs"],
        risk_flags=[],
        provider_name="claude",
        requires_manual_review=False,
        sources=["https://woodland.example/docs/example"],
    )
    base.update(kwargs)
    return SupportInvestigationResult(**base)


def _evaluation(score, verdict="accept", **kwargs) -> DraftConfidenceResult:
    base = dict(
        score=score,
        verdict=verdict,
        summary=f"Judge says {score}",
        missing_information=[],
        investigation_actions=[],
        rewrite_brief="",
        risk_flags=[],
        hard_blockers=[],
        provider="claude",
    )
    base.update(kwargs)
    return DraftConfidenceResult(**base)


def _context(**kwargs) -> SupportPipelineContext:
    base = dict(
        customer_name="Mica",
        subject="Glossary navigation",
        message="The glossary is missing page numbers.",
        theme_tag="trail-atlas",
        knowledge_context="KB context",
        past_tickets_context="Past ticket context",
    )
    base.update(kwargs)
    return SupportPipelineContext(**base)


@pytest.fixture
def pipeline(monkeypatch):
    """First pass scores 5 in manual review; the deep pass and judge are scripted per test."""
    calls = {"deep": [], "collector": []}
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *_, **__: _investigation())
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [Path("/tmp/trail-atlas")])
    scores = {"first": _evaluation(5, "manual_review", missing_information=["Is the taxonomy hierarchical?"]),
              "deep": _evaluation(9, "accept")}

    def judge(*_, draft_html, **__):
        return scores["deep"] if "Deep draft" in draft_html else scores["first"]

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    def deep(request, critique, **kwargs):
        calls["deep"].append({"request": request, "critique": critique, **kwargs})
        return calls.get("deep_result") or _investigation("<p>Deep draft</p>", evidence_summary="register_taxonomy -> hierarchical true")

    monkeypatch.setattr("services.support_pipeline.deep_draft_reply", deep)
    calls["scores"] = scores
    return calls


def test_deep_pass_replaces_a_stuck_draft_and_may_auto_send(pipeline):
    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert len(pipeline["deep"]) == 1
    assert pipeline["deep"][0]["critique"].score == 5
    assert pipeline["deep"][0]["critique"].missing_information == ["Is the taxonomy hierarchical?"]
    assert result.reply_html == "<p>Deep draft</p>"
    assert result.confidence_score == 9
    assert result.delivery_mode == "auto_send"
    assert result.trace[-1]["attempt"] == "deep"
    assert result.trace[-1]["kept"] is True
    assert "Deep pass" in result.decision_note
    assert "5/10 -> 9/10, deep draft kept" in result.decision_note


def test_deep_pass_keeps_the_first_draft_when_it_scores_lower(pipeline):
    pipeline["scores"]["deep"] = _evaluation(3, "manual_review")

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.reply_html == "<p>First draft</p>"
    assert result.confidence_score == 5
    assert result.delivery_mode == "draft"
    assert "first draft kept" in result.decision_note


def test_deep_pass_failure_keeps_the_first_draft(pipeline):
    pipeline["deep_result"] = fallback_investigator_result("claude", "Deep pass failed: timeout")

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.reply_html == "<p>First draft</p>"
    assert "failed" in result.decision_note
    assert result.trace[-1]["error"]


@pytest.mark.parametrize(
    "first",
    [
        _evaluation(9, "accept"),  # already at the bar
        _evaluation(5, "manual_review", risk_flags=["evaluator_failure"]),  # judge broke
    ],
)
def test_deep_pass_does_not_run_when_it_cannot_help(pipeline, first):
    pipeline["scores"]["first"] = first

    run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert pipeline["deep"] == []




def test_deep_pass_off_switch(pipeline):
    run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=OFF)

    assert pipeline["deep"] == []


def test_collected_evidence_reaches_the_deep_request_and_the_workdir_is_removed(pipeline):
    seen = {}

    def collector(evaluation, workdir):
        seen["score"] = evaluation.score
        seen["workdir"] = workdir
        shot = workdir / "screenshot-1.png"
        shot.write_bytes(b"png")
        return DeepEvidence(
            customer_context="Account under other@example.com: Index Book license",
            past_tickets_context="Past Index Book ticket",
            image_paths=[shot],
            notes=["Account found for a named address (other@example.com)."],
        )

    result = run_support_pipeline(
        context=_context(), max_attempts=1, deep_pass_config=ON, deep_evidence=collector
    )

    call = pipeline["deep"][0]
    assert seen["score"] == 5
    assert "Index Book license" in call["request"].customer_context
    assert "Past Index Book ticket" in call["request"].past_tickets_context
    assert call["image_paths"] == [seen["workdir"] / "screenshot-1.png"]
    assert call["workdir"] == seen["workdir"]
    assert call["repo_dirs"] == [Path("/tmp/trail-atlas")]
    assert not seen["workdir"].exists()
    assert result.trace[-1]["images"] == 1


def test_a_failing_collector_does_not_stop_the_deep_pass(pipeline):
    def collector(evaluation, workdir):
        raise TimeoutError("customer endpoint slow")

    result = run_support_pipeline(
        context=_context(), max_attempts=1, deep_pass_config=ON, deep_evidence=collector
    )

    assert len(pipeline["deep"]) == 1
    assert result.reply_html == "<p>Deep draft</p>"
    assert "Evidence collection failed" in result.trace[-1]["evidence_notes"][0]


def test_deep_actions_add_the_baseline_to_the_judges_list():
    actions = deep_actions(_evaluation(5, investigation_actions=["check_woodland-example_repo", "expand_kb_search"]))

    assert actions == ["check_woodland-example_repo", "expand_kb_search", "refresh_customer_context", "expand_past_ticket_search"]


def test_other_customer_emails_skips_the_sender_and_our_addresses(monkeypatch):
    monkeypatch.setattr("services.deep_pass._OWN_DOMAINS", ("woodland.example",))
    message = (
        "Possible email addresses: reader@reading-room.example and maybe editor@reading-room.example "
        "Write to me at Sender@Example.com. Unsubscribe: hello+unsubscribe@woodland.example."
    )

    assert other_customer_emails(message, "sender@example.com") == ["reader@reading-room.example", "editor@reading-room.example"]


def test_deep_draft_reply_runs_read_only_isolated_at_the_configured_effort(monkeypatch, tmp_path):
    captured = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return {"text": _investigation_json(), "total_cost_usd": 0.0, "usage": {}}

    monkeypatch.setattr(support_agent, "invoke_claude_result", fake_invoke)
    monkeypatch.setattr(support_agent, "should_skip_claude", lambda: False)
    repo = tmp_path / "trail-atlas"
    repo.mkdir()
    request = SupportInvestigationRequest(customer_name="Mica", subject="Glossary", message="How are entries numbered?")
    critique = DeepPassCritique(previous_draft_html="<p>First</p>", score=5, summary="unverified",
                                missing_information=["taxonomy hierarchy"])

    result = support_agent.deep_draft_reply(
        request, critique, workdir=tmp_path, repo_dirs=[repo], config=DeepPassConfig(True, "xhigh", 99)
    )

    assert result.model == "opus-deep"
    assert captured["model"] == "opus"
    assert captured["effort"] == "xhigh"
    assert captured["timeout"] == 99
    assert captured["isolated"] is True
    assert captured["cwd"] == tmp_path
    assert set(captured["tools"]) == {"Read", "Grep", "Glob", "WebFetch", "WebSearch"}
    assert f"Read(/{tmp_path.resolve()}/**)" in captured["allowed_tools"]
    assert f"Read(/{repo.resolve()}/**)" in captured["allowed_tools"]
    # Only path-scoped reads: a bare Grep/Glob rule would search the whole machine.
    assert sorted(t for t in captured["allowed_tools"] if not t.startswith("Read(")) == ["WebFetch", "WebSearch"]
    assert "scored 5/10" in captured["prompt"]
    assert "taxonomy hierarchy" in captured["prompt"]
    assert str(repo) in captured["prompt"]
    # Same agent: the same rules file as every first-pass draft.
    assert support_agent.load_rules()[:200] in captured["prompt"]


def test_deep_draft_reply_skips_while_claude_is_down(monkeypatch, tmp_path):
    monkeypatch.setattr(support_agent, "should_skip_claude", lambda: True)
    monkeypatch.setattr(support_agent, "invoke_claude_result", lambda **_: pytest.fail("Claude was called"))
    request = SupportInvestigationRequest(customer_name="Mica", subject="S", message="M")

    result = support_agent.deep_draft_reply(
        request, DeepPassCritique("<p>x</p>", 5, "s"), workdir=tmp_path, config=ON
    )

    assert "investigator_failure" in result.risk_flags


def test_collect_deep_evidence_looks_up_named_addresses_and_saves_screenshots(monkeypatch, tmp_path):
    monkeypatch.setattr("services.deep_pass._OWN_DOMAINS", ("woodland.example",))
    async def fake_investigate(**kwargs):
        assert "refresh_customer_context" in kwargs["evaluation"].investigation_actions
        return {
            "knowledge_context": "Expanded KB",
            "past_tickets_context": "Expanded past tickets",
            "customer_context": "Sender account",
            "additional_context": "Expanded past ticket search.\nAttempt 2 evaluator summary: x",
        }

    lookups = []

    async def fake_customer(email):
        lookups.append(email)
        return {"woocommerce": {"order_count": 1}} if email == "reader@reading-room.example" else None

    monkeypatch.setattr(processor_module, "investigate_ticket_for_confidence", fake_investigate)
    monkeypatch.setattr(processor_module, "fetch_customer_data", fake_customer)

    class FakeHelpScout:
        async def get_attachment_data(self, conversation_id, attachment_id, *, max_bytes):
            assert (conversation_id, attachment_id) == (42, 7)
            return b"\x89PNG", "image/png"

    class FakeProcessor:
        helpscout = FakeHelpScout()

    ticket = {
        "id": 42,
        "subject": "Index Book purchase",
        "message": "Emails: reader@reading-room.example and editor@reading-room.example",
        "customer_email": "sender@example.com",
        "theme_tag": "index-book",
        "attachments": [
            {"id": 7, "filename": "a.png", "mime_type": "image/png", "is_image": True, "is_blank": False},
            {"id": 8, "filename": "blank.png", "mime_type": "image/png", "is_image": True, "is_blank": True},
        ],
    }

    evidence = asyncio.run(processor_module.collect_deep_evidence(
        processor=FakeProcessor(), ticket=ticket, evaluation=_evaluation(5), workdir=tmp_path
    ))

    assert lookups == ["reader@reading-room.example", "editor@reading-room.example"]
    assert "Lookup for reader@reading-room.example" in evidence.customer_context
    assert "Sender account" in evidence.customer_context
    assert "lookup returned nothing for editor@reading-room.example" in evidence.customer_context
    from services.company_pack import wording
    expected = wording("account_unavailable").format(emails="editor@reading-room.example")
    assert expected in evidence.customer_context
    assert evidence.knowledge_context == "Expanded KB"
    assert evidence.image_paths == [tmp_path / "screenshot-1.png"]
    assert (tmp_path / "screenshot-1.png").read_bytes() == b"\x89PNG"
    assert "Expanded past ticket search." in evidence.notes
    assert not any(note.startswith("Attempt ") for note in evidence.notes)


def _investigation_json() -> str:
    import json

    return json.dumps({
        "reply_html": "<p>Deep</p>",
        "confidence_score": 9,
        "decision_summary": "Verified",
        "evidence_summary": "claim -> file:line",
        "investigation_steps": ["Read register_taxonomy"],
        "risk_flags": [],
        "provider_name": "claude",
        "requires_manual_review": False,
        "sources": ["wp-job-manager/includes/class-wp-job-manager-post-types.php"],
    })


def test_cited_source_lines_are_read_by_supa_and_stay_inside_the_checkouts(tmp_path):
    repo = tmp_path / "trail-atlas"
    (repo / "inc" / "integrations").mkdir(parents=True)
    source = "\n".join(f"line {n}" for n in range(1, 121))
    (repo / "inc" / "integrations" / "wp-job-manager.php").write_text(source)
    (tmp_path / "secret.php").write_text("SECRET")
    text = (
        "Labels -> trail-atlas/inc/integrations/wp-job-manager.php:85,88-90. "
        "Menu -> wp-job-manager.php:100 and ../secret.php:1 and /etc/hosts.php:1"
    )

    excerpts = cited_source_excerpts(text, [repo])

    from services.company_pack import wording
    assert wording("source_excerpts") in excerpts
    assert "trail-atlas/inc/integrations/wp-job-manager.php:82-88" in excerpts  # 85 +/- 3
    assert "85: line 85" in excerpts
    assert "88: line 88" in excerpts and "90: line 90" in excerpts
    assert "100: line 100" in excerpts  # bare file name, unique in the checkout
    assert "SECRET" not in excerpts


def test_the_judge_sees_the_cited_lines(pipeline, monkeypatch, tmp_path):
    repo = tmp_path / "trail-atlas"
    repo.mkdir()
    (repo / "types.php").write_text("\n".join(["x"] * 9 + ["'hierarchical' => true,"]))
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [repo])
    pipeline["deep_result"] = _investigation("<p>Deep draft</p>", evidence_summary="Types are hierarchical -> types.php:10")
    judged = []

    def judge(*_, draft_html, knowledge_context, **__):
        judged.append(knowledge_context)
        return _evaluation(9) if "Deep draft" in draft_html else _evaluation(5, "manual_review")

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert "10: 'hierarchical' => true," in judged[-1]
    assert "hierarchical' => true" not in judged[0]
    assert result.trace[-1]["cited_excerpts"] == 1


def test_citations_keep_repo_attribution_for_shared_filenames(tmp_path):
    journal_base, nova = tmp_path / "journal-base", tmp_path / "page-tools"
    journal_base.mkdir()
    nova.mkdir()
    (journal_base / "functions.php").write_text("journal-base search menu\n")
    (nova / "functions.php").write_text("unrelated nova code\n")

    revision = "a1b2c3d4e5f6"
    result = cited_source_excerpts(f"{journal_base.name}@{revision} functions.php:1", [nova, journal_base])

    assert "journal-base search menu" in result
    assert "unrelated nova code" not in result
    assert cited_source_excerpts("functions.php:1", [nova, journal_base]) == ""


def test_local_hash_line_anchors_are_read_as_source_citations(tmp_path):
    repo = tmp_path / "page-tools"
    repo.mkdir()
    (repo / "controls.js").write_text("spacing label\nspacing range\n")

    result = cited_source_excerpts("page-tools/controls.js#L1-L2", [repo])

    assert "1: spacing label" in result
    assert "2: spacing range" in result


def test_repo_qualified_bare_filename_resolves_inside_only_that_checkout(tmp_path):
    journal_base, nova = tmp_path / "journal-base", tmp_path / "page-tools"
    for root in (journal_base, nova):
        (root / "inc").mkdir(parents=True)
    (journal_base / "inc/menus.php").write_text("verified journal-base search suggestions\n")
    (nova / "inc/menus.php").write_text("unrelated nova menus\n")

    result = cited_source_excerpts("Read journal-base menus.php:1", [nova, journal_base])

    assert "verified journal-base search suggestions" in result
    assert "unrelated nova menus" not in result
    assert cited_source_excerpts("Read menus.php:1", [nova, journal_base]) == ""
    (journal_base / "other").mkdir()
    (journal_base / "other/menus.php").write_text("ambiguous journal-base menu\n")
    assert cited_source_excerpts("Read journal-base menus.php:1", [nova, journal_base]) == ""


def test_missing_source_lookup_matches_specific_path_suffix_and_single_control_topic():
    from services.deep_pass import matching_source_citations
    logo = "page-tools packages/blocks/logo/index.js:18-67"
    spacing = "page-tools packages/controls/card-spacing-settings.js:9-118"
    sources = ["page-tools lib/header-nav-projection.php:1-240",
               "page-tools packages/blocks/navigation/index.js:1-150", logo, spacing]

    assert matching_source_citations(
        "Need source for Logo preview (logo/index.js) and Block Top Spacing", sources
    ) == [logo, spacing]


@pytest.mark.parametrize("qualified", [True, False])
def test_bare_filename_lookup_never_follows_a_symlink_outside_checkout(tmp_path, qualified):
    repo = tmp_path / "journal-base"
    (repo / "inc").mkdir(parents=True)
    outside = tmp_path / "outside.php"
    outside.write_text("outside source must stay private\n")
    (repo / "inc/menus.php").symlink_to(outside)

    assert cited_source_excerpts(("journal-base " if qualified else "") + "menus.php:1", [repo]) == ""


def test_long_cited_ranges_keep_the_end_in_bounded_chunks(tmp_path):
    repo = tmp_path / "page-tools"
    repo.mkdir()
    (repo / "controls.js").write_text("\n".join(f"range {n}" for n in range(1, 121)))

    result = cited_source_excerpts("page-tools/controls.js:1-120", [repo])

    assert "120: range 120" in result


def test_excerpt_budget_skips_an_oversized_range_without_losing_later_small_proofs(tmp_path, monkeypatch):
    from services import deep_pass
    repo = tmp_path / "journal-base"
    repo.mkdir()
    (repo / "large.php").write_text("large line " * 100)
    (repo / "small.php").write_text("verified tiny fact\n")
    monkeypatch.setattr(deep_pass, "_EXCERPTS_MAX_CHARS", 100)

    result = cited_source_excerpts("journal-base/large.php:1 journal-base/small.php:1", [repo])

    assert "verified tiny fact" in result


def test_deep_source_reads_satisfy_code_evidence_gate_without_initial_grep_hits(pipeline, monkeypatch, tmp_path):
    repo = tmp_path / "index-book"
    repo.mkdir()
    (repo / "functions.php").write_text("<?php\nwp_enqueue_script('index-book-admin', get_template_directory_uri() . '/admin.js');\n")
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [repo])
    pipeline["deep_result"] = _investigation(
        "<p>Index Book's functions.php loads the admin script from get_template_directory_uri(), so copying the file into a child theme does not replace it.</p>",
        evidence_summary="Parent script URL -> index-book/functions.php:2",
        sources=["index-book/functions.php:2"],
    )
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda *, draft_html, **kw:
        _evaluation(9) if "get_template_directory_uri" in draft_html else _evaluation(5, "retry"))

    result = run_support_pipeline(context=_context(knowledge_context="No initial source hits"),
                                  max_attempts=1, deep_pass_config=ON)

    assert result.delivery_mode == "auto_send"
    assert result.confidence_score == 9
    assert "missing_repo_evidence" not in result.hard_blockers


def test_deep_pass_reads_the_specific_missing_lines_and_rejudges_once(pipeline, monkeypatch, tmp_path):
    repo = tmp_path / "journal-base"
    repo.mkdir()
    (repo / "menus.php").write_text("\n".join(["x"] * 9 + ["register_nav_menu('search-suggestions', 'Search Suggestions');"]))
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [repo])
    pipeline["deep_result"] = _investigation("<p>Deep draft about the search menu heading.</p>")
    deep_contexts = []
    def judge(*, draft_html, knowledge_context, **kw):
        if "Deep draft" not in draft_html:
            return _evaluation(5, "retry")
        deep_contexts.append(knowledge_context)
        if "10: register_nav_menu" in knowledge_context:
            return _evaluation(9)
        return _evaluation(7, "retry", hard_blockers=["missing_repo_evidence"],
                           missing_information=["Need source for Search Suggestions: journal-base/menus.php:10"])
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.delivery_mode == "auto_send"
    assert len(deep_contexts) == 2
    assert result.trace[-1]["evidence_recheck"]["from_score"] == 7


def test_evidence_recheck_uses_prior_citations_when_judge_names_only_the_missing_file(pipeline, monkeypatch, tmp_path):
    from services import deep_pass
    repo = tmp_path / "journal-base"
    repo.mkdir()
    (repo / "large.php").write_text("x" * 200)
    (repo / "menus.php").write_text("\n" * 9 + "register_nav_menu('search-suggestions', 'Search Suggestions');\n")
    monkeypatch.setattr(deep_pass, "_EXCERPTS_MAX_CHARS", 280)
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [repo])
    pipeline["deep_result"] = _investigation("<p>Deep draft about the search menu heading.</p>",
        sources=["journal-base/large.php:1", "journal-base/menus.php:10"])
    def judge(*, draft_html, knowledge_context, **kw):
        if "Deep draft" not in draft_html:
            return _evaluation(5, "retry")
        if "10: register_nav_menu" in knowledge_context:
            return _evaluation(9)
        return _evaluation(7, "retry", hard_blockers=["missing_repo_evidence"],
                           missing_information=["Need source for the Search Suggestions registration in menus.php"])
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.delivery_mode == "auto_send"
    assert result.trace[-1]["evidence_recheck"]["from_score"] == 7


def test_unchanged_recheck_score_preserves_new_evidence_for_the_factual_rewrite(pipeline, monkeypatch, tmp_path):
    repo = tmp_path / "journal-base"
    repo.mkdir()
    (repo / "menus.php").write_text("verified search menu implementation\n")
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [repo])
    pipeline["deep_result"] = _investigation("<p>Deep draft with an unsupported extra claim.</p>")
    requests = []

    def writer(request, **kw):
        requests.append(request)
        if len(requests) == 1:
            return _investigation()
        assert "verified search menu implementation" in request.knowledge_context
        return _investigation("<p>Corrected reply with only the verified search guidance.</p>")

    def judge(*, draft_html, **kw):
        if "Corrected reply" in draft_html:
            assert "verified search menu implementation" in kw["knowledge_context"]
            return _evaluation(8)
        if "Deep draft" in draft_html:
            return _evaluation(7, "retry", hard_blockers=["missing_repo_evidence"],
                               missing_information=["Read journal-base/menus.php:1 and remove the unsupported extra claim"])
        return _evaluation(5, "retry")

    monkeypatch.setattr("services.support_pipeline.draft_reply", writer)
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.trace[-1]["evidence_recheck"]["kept"] is False
    assert result.delivery_mode == "auto_send"
    assert result.trace[-1]["evidence_rewrite"]["kept"] is True


def test_recheck_budget_is_not_consumed_again_by_source_lines_already_supplied(pipeline, monkeypatch, tmp_path):
    from services import deep_pass
    repo = tmp_path / "journal-base"
    repo.mkdir()
    (repo / "large.php").write_text("existing implementation " * 9)
    (repo / "menus.php").write_text("new verified search suggestions implementation\n")
    monkeypatch.setattr(deep_pass, "_EXCERPTS_MAX_CHARS", 300)
    monkeypatch.setattr("services.support_pipeline.repo_dirs_for", lambda **_: [repo])
    pipeline["deep_result"] = _investigation("<p>Deep draft</p>",
        sources=["journal-base/large.php:1", "journal-base/menus.php:1"])

    def judge(*, draft_html, knowledge_context, **kw):
        if "Deep draft" not in draft_html:
            return _evaluation(5, "retry")
        if "new verified search suggestions implementation" in knowledge_context:
            return _evaluation(8)
        assert "existing implementation" in knowledge_context
        return _evaluation(7, "retry", hard_blockers=["missing_repo_evidence"],
                           missing_information=["Need menus.php source; large.php was already supplied"])

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.delivery_mode == "auto_send"


@pytest.mark.parametrize("fixes_css", [True, False])
def test_accepted_deep_reply_gets_one_css_correction_and_every_gate_runs_again(pipeline, monkeypatch, fixes_css):
    bad = '<p>Deep draft.</p><pre>.old-overlay { color: black; }</pre>'
    pipeline["deep_result"] = _investigation(bad)
    writer_calls = []
    def writer(request, **kw):
        writer_calls.append(request)
        if len(writer_calls) == 1:
            return _investigation()
        return _investigation("<p>Corrected reply. Use the verified controls.</p>" if fixes_css else bad)
    monkeypatch.setattr("services.support_pipeline.draft_reply", writer)
    monkeypatch.setattr("services.support_pipeline._detect_dead_css", lambda *, reply_html, **kw:
                        "The .old-overlay class is absent" if ".old-overlay" in reply_html else None)
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda *, draft_html, **kw:
                        _evaluation(8) if "Deep draft" in draft_html or "Corrected reply" in draft_html else _evaluation(5, "retry"))

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert len(writer_calls) == 2
    assert "The .old-overlay class is absent" in writer_calls[-1].internal_note
    if fixes_css:
        assert result.delivery_mode == "auto_send"
        assert result.hard_blockers == []
        assert result.trace[-1]["css_rewrite"]["kept"]
    else:
        assert result.delivery_mode == "draft"
        assert "dead_css_selector" in result.trace[-1]["css_rewrite"]["hard_blockers"]


@pytest.mark.parametrize("fixes_evidence", [True, False])
def test_near_bar_repo_hold_gets_one_factual_rewrite_and_independent_judgment(pipeline, monkeypatch, fixes_evidence):
    bad = "<p>Deep draft with unsupported control ranges.</p>"
    pipeline["deep_result"] = _investigation(bad)
    calls = []
    def writer(request, **kw):
        calls.append(request)
        if len(calls) == 1: return _investigation()
        return _investigation("<p>Verified core guidance. Please send the selected block's name.</p>" if fixes_evidence else bad)
    monkeypatch.setattr("services.support_pipeline.draft_reply", writer)
    def judge(*, draft_html, **kw):
        if "Verified core" in draft_html: return _evaluation(8)
        if "Deep draft" in draft_html:
            return _evaluation(7, "retry", hard_blockers=["missing_repo_evidence"],
                               missing_information=["Source for the exact control ranges is missing"])
        return _evaluation(5, "retry")
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert len(calls) == 2
    assert "Source for the exact control ranges is missing" in calls[-1].internal_note
    assert result.delivery_mode == ("auto_send" if fixes_evidence else "draft")
    if fixes_evidence: assert result.hard_blockers == []
    else: assert "missing_repo_evidence" in result.trace[-1]["evidence_rewrite"]["hard_blockers"]


def test_factual_rewrite_does_not_override_an_explicit_manual_review_verdict(pipeline, monkeypatch):
    pipeline["deep_result"] = _investigation("<p>Deep draft</p>")
    calls = []
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *a, **kw: (calls.append(1) or _investigation()))
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda *, draft_html, **kw:
        _evaluation(8, "manual_review", hard_blockers=["missing_repo_evidence"]) if "Deep draft" in draft_html else _evaluation(5, "retry"))

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert result.delivery_mode == "draft"
    assert len(calls) == 1


def test_the_judge_sees_pinned_github_lines(pipeline, monkeypatch, journal_products):
    """The judge sees verified source lines from a cited branch."""
    from services import github_evidence

    sha = "a1b2c3d4e5f6" + "0" * 28
    monkeypatch.setattr(github_evidence, "_ls_remote", lambda url: {"fix/index-order": sha})
    monkeypatch.setattr(
        github_evidence, "_read_product_file",
        lambda url, ref, path: "\n".join(["x"] * 99 + ["$heading = trim($heading);"]),
    )
    pipeline["deep_result"] = _investigation(
        "<p>Deep draft</p>",
        evidence_summary="trim the chapter heading -> https://github.com/woodland-example/journal-tools/blob/"
                         "fix/index-order/features/metaboxes/init.php#L100",
    )
    judged = []

    def judge(*_, draft_html, knowledge_context, **__):
        judged.append(knowledge_context)
        return _evaluation(9) if "Deep draft" in draft_html else _evaluation(5, "manual_review")

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)

    result = run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=ON)

    assert "Repo evidence (journal-tools):" in judged[-1]
    assert "100: $heading = trim($heading);" in judged[-1]
    assert result.trace[-1]["pinned_github"] == 1


def test_collect_deep_evidence_downloads_pasted_screenshots(monkeypatch, tmp_path, inline_assets):
    async def fake_investigate(**_kwargs):
        return {"customer_context": "", "additional_context": ""}

    monkeypatch.setattr(processor_module, "investigate_ticket_for_confidence", fake_investigate)

    async def no_customer(_email):
        return None

    monkeypatch.setattr(processor_module, "fetch_customer_data", no_customer)
    fetched = []

    class FakeHelpScout:
        async def get_inline_image_data(self, url, *, max_bytes):
            fetched.append(url)
            return b"\x89PNG"

    class FakeProcessor:
        helpscout = FakeHelpScout()

    url = "https://assets.woodland.example/inline/1/a/Screenshot.png"
    ticket = {
        "id": 55, "subject": "Journal index", "message": "Here is the screenshot",
        "customer_email": "", "theme_tag": "index-book",
        "attachments": [
            {"id": None, "url": url, "filename": "Screenshot.png", "mime_type": "image/png",
             "inline": True, "is_image": True, "is_blank": False},
            {"id": None, "url": url + "?logo", "filename": "logo.png", "mime_type": "image/png",
             "inline": True, "is_image": False, "is_signature": True, "is_blank": False},
        ],
    }
    evidence = asyncio.run(processor_module.collect_deep_evidence(
        processor=FakeProcessor(), ticket=ticket, evaluation=_evaluation(5), workdir=tmp_path
    ))
    assert fetched == [url]
    assert evidence.image_paths == [tmp_path / "screenshot-1.png"]


def test_named_products_open_their_own_repository_and_explicit_companion(monkeypatch, tmp_path, journal_products):
    from services import deep_pass

    monkeypatch.setattr(deep_pass, "VALID_SLUGS", {"field-journal", "trail-atlas"})
    assert deep_pass.named_theme_slugs("field-journal and trail-atlas", exclude="field-journal") == ["trail-atlas"]
    assert deep_pass.named_theme_slugs("a field of tall grass") == []
    def checkout(slug):
        directory = tmp_path / slug
        directory.mkdir(exist_ok=True)
        return directory
    monkeypatch.setattr(deep_pass, "ensure_repo_checkout", checkout)
    dirs = deep_pass.repo_dirs_for(theme_tag="trail-atlas", subject="A chapter index", message="",
                                   extra_text="Read Journal Tools before diagnosing this entry.")
    assert [d.name for d in dirs] == ["trail-atlas", "journal-tools"]
    dirs = deep_pass.repo_dirs_for(theme_tag="trail-atlas", subject="", message="", extra_text="an ordinary journal")
    assert [d.name for d in dirs] == ["trail-atlas"]
