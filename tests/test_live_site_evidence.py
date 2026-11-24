from services.live_site_evidence import (
    detect_dead_css_selectors,
    extract_css_class_tokens,
    extract_html_class_names,
    extract_site_url,
    find_missing_classes,
)


LIVE_HEADER_HTML = '<header class="journal-index"><div class="appendix-entry"><span class="entry-marker"></div></header>'

DEAD_DRAFT = """<p>Paste this into Additional CSS:</p>
<p><code>@media (min-width: 720px) {<br />&nbsp;&nbsp;.journal-index .missing-page-number {<br />
&nbsp;&nbsp;&nbsp;&nbsp;display: none;<br />&nbsp;&nbsp;}<br />}</code></p>"""

LIVE_DRAFT = """<p>Paste this into Additional CSS:</p>
<p><code>@media (min-width: 720px) {<br />&nbsp;&nbsp;.appendix-entry .entry-marker {<br />
&nbsp;&nbsp;&nbsp;&nbsp;display: none;<br />&nbsp;&nbsp;}<br />}</code></p>"""


def test_verified_selector_passes_clean():
    assert detect_dead_css_selectors(draft_html=LIVE_DRAFT, site_html=LIVE_HEADER_HTML) is None


def test_prose_before_a_brace_is_not_read_as_a_selector():
    """Regression: the reply body mentions the customer's domain, and its suffix was extracted.

    Line structure is what separates a selector from the sentence in front of it, and
    `strip_html` destroys it by collapsing whitespace.
    """
    draft = """<p>The index for reading-room.example lists both appendix entries.</p>
<p><code>@media (min-width: 720px) {<br />&nbsp;&nbsp;.appendix-entry .entry-marker {<br />
&nbsp;&nbsp;&nbsp;&nbsp;display: none;<br />&nbsp;&nbsp;}<br />}</code></p>"""

    tokens = extract_css_class_tokens(draft)

    assert "example" not in tokens
    assert tokens == ["appendix-entry", "entry-marker"]


def test_reply_without_css_yields_no_tokens_so_the_guard_stays_quiet():
    draft = "<p>Go to Appearance &rarr; Editor and set the Title Size to 60.</p>"

    assert extract_css_class_tokens(draft) == []
    assert detect_dead_css_selectors(draft_html=draft, site_html=LIVE_HEADER_HTML) is None


def test_at_rule_conditions_are_not_selectors():
    draft = "<p><code>@media (min-width: 720px) {<br />&nbsp;&nbsp;.entry-marker { display: none; }<br />}</code></p>"

    assert extract_css_class_tokens(draft) == ["entry-marker"]


def test_unfetchable_site_never_produces_a_verdict():
    """No markup means no knowledge, which must not read as "the selector is missing"."""
    assert detect_dead_css_selectors(draft_html=DEAD_DRAFT, site_html="") is None
    assert find_missing_classes(["missing-page-number"], "") == []


def test_extract_html_class_names_reads_markup_not_stylesheets():
    html = """<style>.only-in-css { color: red; }</style><div class="really-here and-here"></div>"""

    names = extract_html_class_names(html)

    assert names == {"really-here", "and-here"}
    assert "only-in-css" not in names


def test_extract_site_url_skips_owned_and_support_domains(monkeypatch):
    monkeypatch.setattr("services.live_site_evidence.NON_CUSTOMER_DOMAINS", {"woodland.example", "library.example", "sources.example"})
    url = extract_site_url(
        "See https://woodland.example/docs/field-journal/header-and-footer/ for the guide.",
        "My site is https://reading-room.example/about/ if that helps.",
    )

    assert url == "https://reading-room.example/about/"


def test_extract_site_url_returns_none_without_a_customer_domain(monkeypatch):
    monkeypatch.setattr("services.live_site_evidence.NON_CUSTOMER_DOMAINS", {"woodland.example", "library.example", "sources.example"})
    assert extract_site_url("https://library.example/books/", "https://sources.example/woodland-example") is None


# --- pipeline wiring ---------------------------------------------------

import services.support_pipeline as support_pipeline  # noqa: E402
from services.support_pipeline import SupportPipelineContext  # noqa: E402


def _context() -> SupportPipelineContext:
    return SupportPipelineContext(
        customer_name="Reader",
        subject="Field Journal: hide header logo on desktop",
        message="My site is https://reading-room.example/ and the logo still shows.",
        theme_tag="field-journal",
        conversation_history="earlier reply",
        customer_context="",
    )


def test_guard_passes_conversation_text_through_to_the_detector(monkeypatch):
    seen = {}

    def fake_detect(*, draft_html, texts, timeout=10.0):
        seen["draft_html"] = draft_html
        seen["texts"] = list(texts)
        return "draft CSS targets classes that are absent: .missing-page-number"

    monkeypatch.setattr(support_pipeline, "detect_dead_css_selectors_for_context", fake_detect)

    verdict = support_pipeline._detect_dead_css(context=_context(), reply_html=DEAD_DRAFT)

    assert verdict is not None
    assert seen["draft_html"] == DEAD_DRAFT
    assert any("reading-room.example" in text for text in seen["texts"])


def test_guard_is_inconclusive_rather_than_fatal_when_the_check_explodes(monkeypatch):
    """A network wobble must never take down an investigation."""
    def boom(**kwargs):
        raise RuntimeError("DNS failure")

    monkeypatch.setattr(support_pipeline, "detect_dead_css_selectors_for_context", boom)

    assert support_pipeline._detect_dead_css(context=_context(), reply_html=DEAD_DRAFT) is None


def test_helpscout_avatar_cdn_is_not_mistaken_for_the_customer_site():
    """An image address in a thread is never a candidate site."""
    thread = 'A profile picture is available at https://reading-room.example/avatar.png'
    assert extract_site_url(thread) is None


def test_asset_links_are_skipped_even_on_a_customer_domain():
    assert extract_site_url("Logo is at https://reading-room.example/wp-content/uploads/logo.png") is None
    assert extract_site_url("My site is https://reading-room.example/about/") == "https://reading-room.example/about/"


def test_css_guard_checks_the_named_inner_page_even_when_homepage_is_first(monkeypatch):
    from services import live_site_evidence as live

    pages = {
        "https://reading-room.example/": '<body class="home"><p>Home</p></body>',
        "https://reading-room.example/prices/": '<div class="reading-list">Prices</div>',
    }
    monkeypatch.setattr(live, "fetch_site_html", lambda url, **kw: pages[url])

    verdict = live.detect_dead_css_selectors_for_context(
        draft_html='<pre>.reading-list { color: #000; }</pre>',
        texts=["My site is https://reading-room.example/; change the text on https://reading-room.example/prices/"],
    )

    assert verdict is None


def test_css_guard_does_not_combine_classes_from_different_pages(monkeypatch):
    from services import live_site_evidence as live

    pages = {
        "https://reading-room.example/": '<div class="journal-index"></div>',
        "https://reading-room.example/about/": '<div class="missing-page-number"></div>',
    }
    monkeypatch.setattr(live, "fetch_site_html", lambda url, **kw: pages[url])

    assert live.detect_dead_css_selectors_for_context(
        draft_html=DEAD_DRAFT,
        texts=["https://reading-room.example/ and https://reading-room.example/about/"],
    ) is not None


def test_css_guard_is_inconclusive_when_named_inner_page_cannot_be_read(monkeypatch):
    from services import live_site_evidence as live

    monkeypatch.setattr(live, "fetch_site_html", lambda url, **kw: (
        '<body class="home"></body>' if url.endswith(".example/") else ""
    ))

    assert live.detect_dead_css_selectors_for_context(
        draft_html=DEAD_DRAFT,
        texts=["https://reading-room.example/ and https://reading-room.example/private/"],
    ) is None


def test_live_css_grounding_supplies_the_real_page_id_instead_of_an_example(monkeypatch):
    from services import live_site_evidence as live
    monkeypatch.setattr(live, "fetch_site_html", lambda url, **kw:
        '<body class="page page-id-72"><div class="reading-list">Text</div></body>')
    context = SupportPipelineContext(customer_name="C", subject="Prices text", theme_tag="field-journal",
                                     message="Use https://reading-room.example/prices/")
    from agents.support_agent import SupportInvestigationResult
    investigation = SupportInvestigationResult(reply_html='<pre>.page-id-31 .reading-list { color: black; }</pre>',
        confidence_score=8, decision_summary="CSS reply", evidence_summary="", investigation_steps=[],
        risk_flags=[], provider_name="claude", sources=[], requires_manual_review=False)
    monkeypatch.setattr(support_pipeline, "_verify_draft_css", lambda **kw: "")
    monkeypatch.setattr(support_pipeline, "verify_draft_ui_labels", lambda **kw: "")

    grounded = support_pipeline._ground_context_with_draft_evidence(context=context, investigation=investigation)

    assert "https://reading-room.example/prices/" in grounded.knowledge_context
    assert "page-id-72" in grounded.knowledge_context
    assert "page-id-31" in grounded.knowledge_context
    assert "absent" in grounded.knowledge_context


# --- repo-backed CSS verification --------------------------------------

from services.live_site_evidence import (  # noqa: E402
    _source_rank,
    verify_css_classes_in_repo,
    verify_draft_css_against_repo,
)


def test_verification_cites_the_stylesheet_over_a_legacy_bundle():
    """A class lives in many files; only the rule that still runs is worth citing."""
    assert _source_rank("assets/scss/components/_card.scss:119") < _source_rank(
        "assets/js-old/main.js:1055"
    )
    assert _source_rank("style.css:400") < _source_rank("assets/js/app.bundle.js:1079")


def test_unknown_product_tag_yields_no_verification():
    assert verify_css_classes_in_repo(["c-card"], "not-a-repo") == {}
    assert verify_draft_css_against_repo(draft_html=LIVE_DRAFT, product_tag="") == ""


def test_missing_class_is_reported_without_flagging_its_existing_parent():
    verdict = detect_dead_css_selectors(draft_html=DEAD_DRAFT, site_html=LIVE_HEADER_HTML)
    assert verdict.endswith(".missing-page-number")
    assert ".journal-index" not in verdict
