"""Reply formatting stays stable across judging, persistence and Help Scout delivery."""
import pytest

from agents.responder import finalize_response_html
from services.helpscout import ensure_email_paragraph_spacing


SPACER = '<p>&nbsp;</p>'


def test_pipeline_and_processor_finalization_do_not_multiply_blank_lines():
    original = '<p>Hello, Reader!</p><p>First paragraph.</p>' + SPACER + '<p><strong>Section</strong></p><p>Details.</p>'
    once = finalize_response_html(original, '', '')
    twice = finalize_response_html(once, '', '')

    assert twice == once
    assert once.count(SPACER) == 3
    assert ensure_email_paragraph_spacing(twice) == twice


@pytest.mark.parametrize('blank', [SPACER, '<p>\xa0</p>', '<p>&#160;</p>', '<p>&#xA0;</p>', '<p><br /></p>', '<P> </P>'])
def test_existing_blank_paragraph_runs_collapse_to_one(blank):
    original = '<p>First.</p>' + (blank + '\n') * 7 + '<p>Second.</p>'

    formatted = finalize_response_html(original, '', '')

    assert formatted.count(SPACER) == 1
    assert finalize_response_html(formatted, '', '') == formatted
    assert 'First.' in formatted and 'Second.' in formatted


def test_one_existing_spacer_does_not_hide_other_unspaced_boundaries():
    original = '<p>A.</p>' + SPACER + '<p>B.</p><ol><li>One</li></ol><p>C.</p>'

    formatted = ensure_email_paragraph_spacing(original)

    assert formatted.count(SPACER) == 3
    assert ensure_email_paragraph_spacing(formatted) == formatted
    assert '<ol><li>One</li></ol>' in formatted


@pytest.mark.parametrize('tag', ['pre', 'code', 'table', 'blockquote'])
def test_verbatim_code_tables_and_quoted_html_keep_their_internal_whitespace(tag):
    protected = f'<{tag}><p>First&nbsp;value</p>\n{SPACER}{SPACER}<p>Second  value</p></{tag}>'
    original = '<p>Explanation.</p>' + protected + '<p>Done.</p>'

    formatted = finalize_response_html(original, '', '')

    assert protected in formatted
    assert finalize_response_html(formatted, '', '') == formatted


def test_inline_nonbreaking_spaces_and_br_style_replies_are_preserved():
    original = 'Hello, Reader!<br><br>Use&nbsp;this.<br><br>Thank you,<br>Support'

    assert finalize_response_html(original, '', '') == original
    assert ensure_email_paragraph_spacing(original) == original


def test_nested_quoted_threads_remain_byte_for_byte_unchanged():
    quote = '<blockquote><blockquote><p>Nested quote.</p></blockquote><p>Outer quote.</p>' + SPACER * 3 + '<p>Still quoted.</p></blockquote>'

    assert quote in finalize_response_html('<p>Reply.</p>' + quote, '', '')


def test_blank_paragraph_anchor_is_not_discarded():
    anchor = '<p id="details">&nbsp;</p>'

    assert anchor in finalize_response_html('<p>Read below.</p>' + anchor + '<p>Details.</p>', '', '')


@pytest.mark.parametrize('comment', ['<!-- note -- >', '<!note>', '<![unknown]>'])
def test_lenient_or_malformed_html_does_not_break_reply_formatting(comment):
    original = '<p>Reply.</p>' + comment + '<p>Details.</p>'

    formatted = finalize_response_html(original, '', '')

    assert comment in formatted
    assert finalize_response_html(formatted, '', '') == formatted
