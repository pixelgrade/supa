import re
from pathlib import Path
from typing import Dict, Any, List
import yaml


def parse_markdown_file(file_path: Path) -> Dict[str, Any]:
    """Parse a markdown knowledge file with YAML frontmatter."""
    content = file_path.read_text(encoding="utf-8")

    # Extract frontmatter
    frontmatter = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                pass
            body = parts[2].strip()

    # Extract title from first heading
    title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    title = title_match.group(1) if title_match else file_path.stem

    return {
        "id": file_path.stem,
        "title": title,
        "tags": frontmatter.get("tags", []),
        "theme": frontmatter.get("theme", "all"),
        "content": body,
        "source": "examples",
        "file_path": str(file_path)
    }


def load_knowledge_files(directory: Path) -> List[Dict[str, Any]]:
    """Load all markdown files from a directory."""
    results = []

    if not directory.exists():
        return results

    for md_file in directory.glob("*.md"):
        try:
            parsed = parse_markdown_file(md_file)
            results.append(parsed)
        except Exception as e:
            print(f"Error parsing {md_file}: {e}")

    return results


def get_all_knowledge(knowledge_dir: Path) -> List[Dict[str, Any]]:
    """Load all knowledge from examples and internal directories."""
    all_docs = []

    examples_dir = knowledge_dir / "examples"
    internal_dir = knowledge_dir / "internal"

    for doc in load_knowledge_files(examples_dir):
        doc["source"] = "examples"
        doc["priority"] = 10  # Highest priority
        all_docs.append(doc)

    for doc in load_knowledge_files(internal_dir):
        doc["source"] = "internal"
        doc["priority"] = 8
        all_docs.append(doc)

    return all_docs
