import pytest
from pathlib import Path
from services.knowledge import load_knowledge_files, parse_markdown_file


def test_parse_markdown_file(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("""---
tags: [refund, billing]
theme: all
---

# Refund request

## Problem
Customer wants refund.

## Example reply
Hello, thanks for reaching out.
""")

    result = parse_markdown_file(md_file)

    assert result["title"] == "Refund request"
    assert "refund" in result["tags"]
    assert result["theme"] == "all"
    assert "Customer wants refund" in result["content"]


def test_load_knowledge_files(tmp_path):
    # Create examples dir
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    (examples_dir / "test1.md").write_text("""---
tags: [tag1]
theme: field-journal
---
# Test 1
Content 1
""")

    (examples_dir / "test2.md").write_text("""---
tags: [tag2]
theme: trail-atlas
---
# Test 2
Content 2
""")

    results = load_knowledge_files(examples_dir)

    assert len(results) == 2
    assert any(r["title"] == "Test 1" for r in results)
    assert any(r["title"] == "Test 2" for r in results)
