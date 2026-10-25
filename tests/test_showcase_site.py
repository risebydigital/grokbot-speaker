"""Guards for the Milo showcase site and the Pages workflow."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

FORBIDDEN = re.compile(r"\b(?:free|founder|senior)\b|www\.|#008073", re.IGNORECASE)
OPACITY = re.compile(r"opacity\s*:")
URL = re.compile(r"https?://[^\s\"')]+")
ALLOWED_URLS = {"https://risebydigital.com"}
ALLOWED_AT = {"milo@bookshelf", "git@github.com"}


def _text_files() -> list[Path]:
    files = [path for path in SITE.rglob("*") if path.is_file() and path.suffix in {".html", ".css", ".js", ".svg"}]
    files.append(ROOT / ".github" / "workflows" / "pages.yml")
    return files


def test_pages_workflow_is_manual_only() -> None:
    text = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "workflow_dispatch" in text
    assert "push:" not in text
    assert "pull_request:" not in text
    assert "path: site" in text


def test_showcase_copy_rules() -> None:
    for path in _text_files():
        text = path.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        assert FORBIDDEN.search(text) is None, path
        assert OPACITY.search(text) is None, path
        for url in URL.findall(text):
            url = url.rstrip(".,)")
            assert url in ALLOWED_URLS, (path, url)


def test_site_has_no_third_party_calls() -> None:
    script = (SITE / "js" / "site.js").read_text(encoding="utf-8")
    assert "fetch(" not in script
    assert "XMLHttpRequest" not in script
    assert "WebSocket" not in script
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert "google-analytics" not in html
    assert "fonts.googleapis.com" not in html
    assert 'src="js/site.js"' in html
    assert 'href="css/site.css"' in html


def test_placeholders_are_marked() -> None:
    script = (SITE / "js" / "site.js").read_text(encoding="utf-8")
    assert script.count("PLACEHOLDER awaiting Adam") >= 3
    assert "items from the repo roadmap - to be confirmed" in script
    assert "credits" in script


def test_readme_banner_points_at_approved_art() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'alt="Milo. Your voice assistant on the bookshelf."' in readme
    assert "docs/assets/banner.svg" in readme
    assert "docs/assets/banner.png" in readme
    assert "docs/assets/banner@2x.png" in readme
    banner = (ROOT / "docs" / "assets" / "banner.svg").read_text(encoding="utf-8")
    assert "#00A897" in banner
    assert "Grok Voice" not in banner
    assert (ROOT / "docs" / "assets" / "social-preview.png").is_file()
    assert (ROOT / "docs" / "assets" / "brand" / "milo-A-mascot-transparent.svg").is_file()
    assert (ROOT / "docs" / "assets" / "brand" / "milo-A-favicon-navy.ico").is_file()
