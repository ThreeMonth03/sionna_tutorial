"""Ask GitHub's Markdown API to verify every documented math expression."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    *sorted((ROOT / "docs").rglob("*.md")),
]
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?:\\.|[^$\n])*(?<!\\)\$")


def expected_math_expression_count(text: str) -> int:
    """Count fenced display math and inline math outside ordinary code fences."""

    count = 0
    fence_kind: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if fence_kind is None:
                fence_kind = stripped[3:].strip() or "code"
                if fence_kind == "math":
                    count += 1
            elif stripped == "```":
                fence_kind = None
            continue

        if fence_kind is None:
            without_code = INLINE_CODE_RE.sub("", line)
            count += len(INLINE_MATH_RE.findall(without_code))

    return count


def render_with_github(markdown: str) -> str:
    """Render Markdown with the same public GitHub REST endpoint used for GFM."""

    payload = json.dumps(
        {
            "text": markdown,
            "mode": "gfm",
            "context": "ThreeMonth03/sionna_tutorial",
        }
    ).encode("utf-8")
    headers = {
        "Accept": "text/html",
        "Content-Type": "application/json",
        "User-Agent": "sionna-tutorial-math-check",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        "https://api.github.com/markdown",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> None:
    failures: list[str] = []

    for path in MARKDOWN_FILES:
        markdown = path.read_text(encoding="utf-8")
        expected = expected_math_expression_count(markdown)
        rendered = render_with_github(markdown)
        actual = rendered.count("<math-renderer")

        if actual != expected:
            failures.append(
                f"{path.relative_to(ROOT)}: expected {expected} rendered math "
                f"expressions, GitHub returned {actual}"
            )

    if failures:
        raise SystemExit("GitHub Markdown math rendering failed:\n" + "\n".join(failures))

    print("GitHub rendered every inline and display math expression successfully.")


if __name__ == "__main__":
    main()
