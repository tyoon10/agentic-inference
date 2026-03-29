"""
News aggregator tools for the agent loop.

Provides RSS fetching, article extraction, and date filtering tools
that the Mistral Small 4 agent uses to build a weekly AI news digest.
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from html import unescape

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "projects" / "01_tool_calling"))
from tools import ToolRegistry


registry = ToolRegistry()

# ── RSS / news sources ──

AI_FEEDS = {
    "openai_blog": "https://openai.com/blog/rss.xml",
    "google_ai_blog": "https://blog.google/technology/ai/rss/",
    "mit_tech_ai": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "arxiv_cs_ai": "http://export.arxiv.org/rss/cs.AI",
    "huggingface_blog": "https://huggingface.co/blog/feed.xml",
    "anthropic_news": "https://www.anthropic.com/rss.xml",
    "nvidia_blog": "https://blogs.nvidia.com/feed/",
    "deepmind_blog": "https://deepmind.google/blog/rss.xml",
    "mistral_news": "https://mistral.ai/news/rss.xml",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AINewsAgent/1.0)"
}


def _fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch URL content with timeout and error handling."""
    req = Request(url, headers=HEADERS)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except URLError as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def _parse_rss(xml_text: str) -> list[dict]:
    """Parse RSS/Atom XML into a list of article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # Handle RSS 2.0
    for item in root.iter("item"):
        articles.append(_extract_rss_item(item))

    # Handle Atom feeds
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        articles.append(_extract_atom_entry(entry, ns))

    return articles


def _extract_rss_item(item) -> dict:
    """Extract fields from an RSS <item>."""
    title = item.findtext("title", "").strip()
    link = item.findtext("link", "").strip()
    pub_date = item.findtext("pubDate", "").strip()
    description = item.findtext("description", "").strip()
    # Strip HTML tags from description
    description = re.sub(r"<[^>]+>", "", unescape(description))[:500]
    return {
        "title": title,
        "link": link,
        "published": pub_date,
        "summary": description,
    }


def _extract_atom_entry(entry, ns) -> dict:
    """Extract fields from an Atom <entry>."""
    title = entry.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
    link_el = entry.find("{http://www.w3.org/2005/Atom}link")
    link = link_el.get("href", "") if link_el is not None else ""
    published = entry.findtext("{http://www.w3.org/2005/Atom}published", "")
    updated = entry.findtext("{http://www.w3.org/2005/Atom}updated", "")
    summary_el = entry.findtext("{http://www.w3.org/2005/Atom}summary", "")
    content_el = entry.findtext("{http://www.w3.org/2005/Atom}content", "")
    desc = summary_el or content_el
    desc = re.sub(r"<[^>]+>", "", unescape(desc))[:500]
    return {
        "title": title,
        "link": link,
        "published": published or updated,
        "summary": desc,
    }


# ── Registered tools ──

@registry.register
def list_news_sources() -> str:
    """List all available AI news RSS feed sources."""
    return json.dumps({
        "sources": list(AI_FEEDS.keys()),
        "count": len(AI_FEEDS),
    })


@registry.register
def fetch_feed(source_name: str) -> str:
    """Fetch articles from a named AI news source. Use list_news_sources to see available names."""
    if source_name not in AI_FEEDS:
        return json.dumps({"error": f"Unknown source: {source_name}. Use list_news_sources."})

    url = AI_FEEDS[source_name]
    try:
        xml_text = _fetch_url(url)
        articles = _parse_rss(xml_text)
        return json.dumps({
            "source": source_name,
            "url": url,
            "articles": articles[:20],
            "total_found": len(articles),
        })
    except RuntimeError as e:
        return json.dumps({"source": source_name, "error": str(e)})


@registry.register
def fetch_article_text(url: str) -> str:
    """Fetch the full text content of an article URL. Returns extracted text (HTML stripped)."""
    try:
        html = _fetch_url(url, timeout=20)
        # Simple content extraction: find <article> or <main>, fall back to <body>
        for tag in ["article", "main", "body"]:
            match = re.search(
                rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE
            )
            if match:
                html = match.group(1)
                break
        # Strip tags, collapse whitespace
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 5000:
            text = text[:5000] + f"... [truncated, {len(text)} chars total]"
        return json.dumps({"url": url, "text": text, "length": len(text)})
    except RuntimeError as e:
        return json.dumps({"url": url, "error": str(e)})


@registry.register
def get_date_range(days_back: int = 7) -> str:
    """Get the date range for filtering articles. Returns start and end dates for the past N days."""
    end = datetime.now()
    start = end - timedelta(days=days_back)
    return json.dumps({
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "days": days_back,
    })


@registry.register
def save_digest(content: str, filename: str = "weekly-digest.md") -> str:
    """Save the compiled news digest to a markdown file in the output directory."""
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(content, encoding="utf-8")
    return json.dumps({"saved": str(path), "size": len(content)})
