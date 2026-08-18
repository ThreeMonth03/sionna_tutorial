import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    *sorted((ROOT / "docs").rglob("*.md")),
]

INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?:\\.|[^$\n])*(?<!\\)\$")
LATEX_COMMAND_RE = re.compile(
    r"\\(?:"
    r"begin|end|mathbf|mathrm|widehat|hat|frac|sqrt|sum|int|delta|"
    r"alpha|beta|gamma|kappa|lambda|phi|theta|tau|pi|pm|ldots|"
    r"sin|cos|exp|left|right|times|cdot|in|leq|geq"
    r")\b"
)
PSEUDO_MATH_RE = re.compile(r"\b(?:H_hat|X_hat|h\(t,\s*tau\)|H\(t,\s*tau\))\b")
UNSUPPORTED_DELIMITERS = (r"\[", r"\]", r"\(", r"\)")


def _unescaped_dollar_count(text: str) -> int:
    return len(re.findall(r"(?<!\\)\$", text))


def test_markdown_math_is_github_renderable() -> None:
    """Keep every mathematical expression inside GitHub-supported syntax."""

    violations: list[str] = []

    for path in MARKDOWN_FILES:
        relative = path.relative_to(ROOT)
        fence_kind: str | None = None
        math_body_lines = 0

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if stripped.startswith("```"):
                if fence_kind is None:
                    fence_kind = stripped[3:].strip() or "code"
                    math_body_lines = 0
                elif stripped == "```":
                    if fence_kind == "math" and math_body_lines == 0:
                        violations.append(f"{relative}:{line_number}: empty math fence")
                    fence_kind = None
                continue

            if fence_kind is not None:
                if fence_kind == "math" and stripped:
                    math_body_lines += 1
                continue

            without_code = INLINE_CODE_RE.sub("", line)

            for delimiter in UNSUPPORTED_DELIMITERS:
                if delimiter in without_code:
                    violations.append(
                        f"{relative}:{line_number}: unsupported delimiter {delimiter!r}"
                    )

            if "$$" in without_code:
                violations.append(
                    f"{relative}:{line_number}: use a fenced ```math block instead of $$"
                )

            if _unescaped_dollar_count(without_code) % 2:
                violations.append(f"{relative}:{line_number}: unbalanced inline $ delimiter")
                continue

            outside_inline_math = INLINE_MATH_RE.sub("", without_code)
            if LATEX_COMMAND_RE.search(outside_inline_math):
                violations.append(
                    f"{relative}:{line_number}: LaTeX command outside a math region"
                )
            if PSEUDO_MATH_RE.search(outside_inline_math):
                violations.append(
                    f"{relative}:{line_number}: pseudo-math text should use inline LaTeX"
                )

        if fence_kind is not None:
            violations.append(f"{relative}: unclosed `{fence_kind}` code fence")

    assert not violations, "GitHub-incompatible Markdown math:\n" + "\n".join(violations)
