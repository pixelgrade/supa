import pytest


@pytest.mark.parametrize("left,right", [
    ("<p>Project &raquo; Content Builder</p>", "<p>Project » Content Builder</p>"),
    ('<p title="A &amp; B">&#187;</p>', '<p title="A & B">»</p>'),
])
def test_email_html_compares_entity_spellings_without_changing_structure(left, right):
    from services.email_html import email_html_equivalent
    assert email_html_equivalent(left, right)


@pytest.mark.parametrize("left,right", [
    ("<p>&lt;b&gt;Text&lt;/b&gt;</p>", "<p><b>Text</b></p>"),
    ("<p>&amp;raquo;</p>", "<p>»</p>"),
    ('<a href="https://example.com/a">Link</a>', '<a href="https://example.com/b">Link</a>'),
    ("<p>Text</p><!--reviewed-->", "<p>Text</p>"),
    ("<p>Two words</p>", "<p>Two  words</p>"),
    ('<a href="a" href="b">Link</a>', '<a href="b" href="a">Link</a>'),
])
def test_email_html_retains_real_text_markup_attribute_comment_and_space_edits(left, right):
    from services.email_html import email_html_equivalent
    assert not email_html_equivalent(left, right)
