from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def test_markdown_uses_github_math_delimiters() -> None:
    """Reject LaTeX delimiters that GitHub Markdown does not render as math."""

    violations: list[str] = []
    for path in MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped in {r"\[", r"\]"}:
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {stripped}")

        # Inline math on GitHub should use $...$, not LaTeX \(...\).
        if r"\(" in text or r"\)" in text:
            violations.append(f"{path.relative_to(ROOT)}: contains \\(...\\) inline math")

        # Every display-math block contributes two $$ delimiters.
        if text.count("$$") % 2:
            violations.append(f"{path.relative_to(ROOT)}: unbalanced $$ delimiter")

    assert not violations, "GitHub-incompatible Markdown math:\n" + "\n".join(violations)
