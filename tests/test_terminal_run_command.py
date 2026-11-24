from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"
INDEX_HTML = Path(__file__).resolve().parents[1] / "static" / "index.html"


def test_terminal_run_command_is_primary_surface():
    app_js = APP_JS.read_text()
    index_html = INDEX_HTML.read_text()

    assert 'help-cmd">run &lt;id&gt;</span> Run a HelpScout ticket' in app_js
    assert "case 'run':" in app_js
    assert "case 'test':" in app_js
    assert 'fetch(\'/dashboard/test-ticket\'' in app_js
    assert 'Run "run <id>" first.' in app_js
    assert 'placeholder="__RUN_HINT__"' in index_html


def test_dashboard_serves_configured_source_at_both_urls(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app

    monkeypatch.setenv("SUPA_SOURCE_URL", "https://source.example.org/desk?one=1&two=2")
    with_source = 'href="https://source.example.org/desk?one=1&amp;two=2"'
    client = TestClient(app)
    for url in ("/", "/static/index.html"):
        page = client.get(url)
        assert page.status_code == 200
        assert with_source in page.text and 'placeholder="run ' in page.text
