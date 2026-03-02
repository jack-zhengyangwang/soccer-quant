"""Qualitative news layer — Tavily search + Claude summarisation."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "news_cache"
_CACHE_TTL_HOURS = 24

# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_key(home: str, away: str) -> str:
    """Deterministic filename for a matchup on today's date."""
    slug = f"{home}_{away}".lower().replace(" ", "_")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{slug}_{date_str}.json"


def _read_cache(home: str, away: str) -> dict | None:
    """Return cached dict if it exists and is younger than TTL, else None."""
    path = _CACHE_DIR / _cache_key(home, away)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        fetched = datetime.fromisoformat(data["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        if age_hours < _CACHE_TTL_HOURS:
            data["source"] = "cache"
            return data
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _write_cache(home: str, away: str, data: dict) -> None:
    """Persist result dict to the cache directory."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / _cache_key(home, away)
    path.write_text(json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# Search layer (Tavily)
# ---------------------------------------------------------------------------


def _tavily_client():
    """Lazy-import and return a TavilyClient, or None if unavailable."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except Exception:
        return None


def _search_team_news(home: str, away: str) -> list[dict]:
    """General web search for team news around this matchup."""
    client = _tavily_client()
    if client is None:
        return []
    try:
        resp = client.search(
            query=f"{home} vs {away} Premier League team news preview",
            topic="general",
            max_results=5,
        )
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in resp.get("results", [])
        ]
    except Exception as exc:
        log.warning("Tavily team-news search failed: %s", exc)
        return []


def _search_match_reports(home: str, away: str) -> list[dict]:
    """News-focused search for injuries, lineups, tactical updates."""
    client = _tavily_client()
    if client is None:
        return []
    try:
        resp = client.search(
            query=f"{home} {away} injury lineup team news",
            topic="news",
            max_results=5,
        )
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in resp.get("results", [])
        ]
    except Exception as exc:
        log.warning("Tavily match-reports search failed: %s", exc)
        return []


def _deduplicate(results: list[dict]) -> list[dict]:
    """Remove duplicate results by URL."""
    seen: set[str] = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# LLM layer (Claude)
# ---------------------------------------------------------------------------


def _summarize_with_claude(
    home: str, away: str, search_results: list[dict]
) -> list[dict]:
    """Send search results to Claude and get structured bullet summaries."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not search_results:
        return _fallback_bullets(search_results)

    try:
        import anthropic
    except ImportError:
        return _fallback_bullets(search_results)

    # Build numbered article list with URLs for source referencing
    articles = "\n\n".join(
        f"[{i+1}] {r['title']}\nURL: {r['url']}\n{r['content'][:600]}"
        for i, r in enumerate(search_results)
    )

    prompt = (
        f"You are a football analyst. Analyse the following search results about "
        f"the upcoming {home} vs {away} Premier League match.\n\n"
        f"Return a JSON object with exactly 4 keys, each containing an array of bullets:\n\n"
        f'{{"injury": [...], "team_report": [...], "sentiment": [...], "tactical": [...]}}\n\n'
        f"Each bullet must have:\n"
        f'  "headline": short one-line summary (≤15 words)\n'
        f'  "detail": 2-3 sentence elaboration with specific facts\n'
        f'  "sources": array of {{"title": "...", "url": "..."}} objects referencing '
        f"which article(s) support this bullet (use the URLs provided)\n\n"
        f"Section guidelines:\n"
        f'- "injury": injury updates, fitness doubts, confirmed absentees, return dates\n'
        f'- "team_report": predicted lineups, squad news, manager quotes, selection dilemmas\n'
        f'- "sentiment": fan/pundit opinions, betting odds shifts, confidence levels, '
        f"form momentum\n"
        f'- "tactical": formation changes, tactical previews, key battles, style matchups\n\n'
        f"Each section should have 1-3 bullets. If no info exists for a section, "
        f"return an empty array.\n\n"
        f"Articles:\n{articles}\n\n"
        f"Respond with ONLY the JSON object, no markdown fences."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        sections = json.loads(text)
        if isinstance(sections, dict):
            # Validate structure
            for key in ("injury", "team_report", "sentiment", "tactical"):
                if key not in sections:
                    sections[key] = []
            return sections
    except Exception as exc:
        log.warning("Claude summarisation failed: %s", exc)

    return _fallback_sections(search_results)


_CATEGORY_KEYWORDS = {
    "injury": [
        "injur", "fitness", "doubt", "absent", "miss", "hamstring", "knee",
        "ankle", "groin", "muscle", "sideline", "ruled out", "return date",
        "scan", "surgery", "rehab", "setback", "knock", "strain", "ligament",
        "suspend", "ban", "red card", "missing player",
    ],
    "sentiment": [
        "odds", "predict", "bet", "favourite", "underdog", "confidence",
        "expect", "tipster", "probability", "forecast", "pick", "wager",
        "pundit", "verdict", "momentum", "fan", "opinion", "poll",
    ],
    "tactical": [
        "formation", "tactic", "lineup", "line-up", "system", "shape",
        "pressing", "counter", "wing", "midfield battle", "key battle",
        "matchup", "style", "approach", "strategy", "switch", "rotation",
    ],
}


def _classify_article(title: str, content: str) -> str:
    """Classify an article into a section using keyword matching."""
    text = f"{title} {content}".lower()
    scores = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "team_report"


def _fallback_sections(search_results: list[dict]) -> dict:
    """Distribute articles across sections using keyword classification."""
    sections = {"injury": [], "team_report": [], "sentiment": [], "tactical": []}

    for r in search_results[:8]:
        title = r.get("title", "")
        content = r.get("content", "")
        cat = _classify_article(title, content)
        sections[cat].append({
            "headline": title[:80],
            "detail": content[:250] if content else "",
            "sources": [{"title": title[:60], "url": r["url"]}],
        })

    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_match_news(home: str, away: str) -> dict:
    """Main entry point — returns structured news sections for a matchup.

    Returns:
        {
            "sections": {
                "injury": [{"headline", "detail", "sources": [{"title", "url"}]}],
                "team_report": [...],
                "sentiment": [...],
                "tactical": [...],
            },
            "fetched_at": "ISO timestamp",
            "source": "live" | "cache" | "unavailable",
        }
    """
    # 1. Cache check
    cached = _read_cache(home, away)
    if cached is not None:
        return cached

    # 2. Search
    general = _search_team_news(home, away)
    news = _search_match_reports(home, away)
    combined = _deduplicate(general + news)

    if not combined:
        return {
            "sections": {},
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "unavailable",
        }

    # 3. Summarise
    sections = _summarize_with_claude(home, away, combined)

    result = {
        "sections": sections,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "live",
    }

    # 4. Cache write
    _write_cache(home, away, result)

    return result


_HIGHLIGHT_ICONS = {
    "injury": "\U0001f3e5",
    "team_report": "\U0001f4cb",
    "sentiment": "\U0001f4ca",
    "tactical": "\u2694\ufe0f",
}


def get_news_highlights(news_data: dict, max_items: int = 4) -> list[str]:
    """Extract the top headline from each non-empty news section as insight bullets.

    Returns a list of short strings like:
        ["🏥 Julio Soler ruled out with hamstring injury", ...]
    """
    sections = news_data.get("sections", {})
    if not sections:
        return []

    highlights = []
    # Prioritise: injury first, then tactical, sentiment, team_report
    for key in ("injury", "tactical", "sentiment", "team_report"):
        bullets = sections.get(key, [])
        for b in bullets:
            headline = b.get("headline", "").strip()
            if headline:
                icon = _HIGHLIGHT_ICONS.get(key, "")
                highlights.append(f"{icon} {headline}")
                if len(highlights) >= max_items:
                    return highlights
    return highlights
