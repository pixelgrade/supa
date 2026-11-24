import pytest
from services.crawler import extract_text_from_html, parse_docs_page


def test_extract_text_from_html():
    html = """
    <html>
    <body>
        <h1>Getting Started</h1>
        <p>This is a guide to help you.</p>
        <ul>
            <li>Step 1</li>
            <li>Step 2</li>
        </ul>
    </body>
    </html>
    """

    text = extract_text_from_html(html)

    assert "Getting Started" in text
    assert "guide to help" in text
    assert "Step 1" in text


def test_parse_docs_page():
    html = """
    <html>
    <head><title>Header Setup - Field Journal</title></head>
    <body>
        <article>
            <h1>Header Setup</h1>
            <p>Learn how to configure your header.</p>
        </article>
    </body>
    </html>
    """

    result = parse_docs_page(html, "https://woodland.example/docs/field-journal/header/")

    assert result["title"] == "Header Setup"
    assert "configure your header" in result["content"]
    assert result["theme"] == "field-journal"
