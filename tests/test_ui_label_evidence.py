from services.ui_label_evidence import (
    extract_ui_label_claims,
    verify_draft_ui_labels,
    verify_ui_labels_in_repos,
)


INDEX_DRAFT = '<p>Open the "Index Layout" section to find the "Marker Width" option.</p>'


def test_core_wordpress_labels_are_never_claims():
    draft = (
        "<p>Go to <strong>Appearance → Customize → Additional CSS</strong>, paste it in, "
        'and click "Save Changes". Then check <strong>Settings → Permalinks</strong>.</p>'
    )

    assert extract_ui_label_claims(draft) == []


def test_bolded_labels_and_breadcrumb_segments_are_extracted():
    draft = (
        "<p>That goes in Appearance → Editor → <strong>Index Layout</strong>, where "
        "you will find <strong>Marker Width</strong> and drag the slider.</p>"
    )

    labels = extract_ui_label_claims(draft)

    assert "Index Layout" in labels
    assert "Marker Width" in labels
    # Prose around the breadcrumb must not survive as a "label".
    assert all(not label.lower().startswith(("that", "go ")) for label in labels)


def test_prose_headings_and_error_messages_are_not_labels():
    draft = (
        "<p><strong>What is actually happening</strong></p>"
        '<p>The page shows "page not found" until you reload.</p>'
    )

    assert extract_ui_label_claims(draft) == []


def test_unreachable_repos_degrade_to_nothing():
    assert verify_ui_labels_in_repos(["Index Layout"], ["not-a-repo"]) == ({}, [])
    assert verify_draft_ui_labels(draft_html="<p>x</p>", product_tag="not-a-repo") == ""


# --- pipeline grounding wiring -----------------------------------------

from services.support_pipeline import (  # noqa: E402
    SupportPipelineContext,
    _ground_context_with_draft_evidence,
)


class _FakeInvestigation:
    reply_html = INDEX_DRAFT


def _context(**overrides) -> SupportPipelineContext:
    values = dict(
        customer_name="Reader",
        subject="Chapter numbering",
        message="Can I change the index markers?",
        theme_tag="field-journal",
        knowledge_context="existing knowledge",
    )
    values.update(overrides)
    return SupportPipelineContext(**values)


def test_grounding_returns_the_context_unchanged_when_there_is_nothing_to_verify(monkeypatch):
    def boom(tag, refresh=False):
        raise RuntimeError("offline")

    monkeypatch.setattr("services.repo_evidence.ensure_repo_checkout", boom)

    context = _context()
    investigation = _FakeInvestigation()

    grounded = _ground_context_with_draft_evidence(context=context, investigation=investigation)

    assert grounded is context


def test_only_labels_present_in_a_readable_checkout_become_evidence(monkeypatch, tmp_path):
    from services import repo_catalog, repo_evidence

    monkeypatch.setattr(repo_catalog, "is_product_repo", lambda tag: tag == "journal")
    monkeypatch.setattr(repo_evidence, "repo_url", lambda tag: "https://source.example/journal")
    monkeypatch.setattr(repo_evidence, "companion_repo_targets", lambda *args, **kwargs: [])
    monkeypatch.setattr(repo_evidence, "ensure_repo_checkout", lambda tag: tmp_path)
    (tmp_path / "inspector.js").write_text('const label = "Index Layout";\n')
    result = verify_draft_ui_labels(draft_html=INDEX_DRAFT, product_tag="journal")
    assert "Repo evidence (journal)" in result and "journal inspector.js:1" in result
    assert "Marker Width" in result and "NOT found" in result
    result = verify_draft_ui_labels(draft_html=INDEX_DRAFT, product_tag="journal", context_text="Marker Width")
    assert "Marker Width" not in result
