from services import theme_requirements


def test_late_requirements_remain_visible_past_the_excerpt(monkeypatch):
    monkeypatch.setattr(theme_requirements, "REQUIREMENT_SNIPPET_PATTERNS",
                        (r"Index format version \d+",))
    text = ("Introduction to chapter numbering. " * 100) + "Index format version 7."
    rendered = theme_requirements.format_theme_requirements_context({
        "title": "Journal index", "url": "https://woodland.example/docs/index/",
        "content": text,
    })
    requirements, excerpt = rendered.split("Excerpt:")
    assert "- Index format version 7." in requirements
    assert "Index format" not in excerpt
