"""Insert missing digit/Cyrillic spaces in Markdown prose, preserving code.

The cleaner never changes digits or punctuation. Latin identifiers (including
h1, SHA256 and underscore names), code, math, URLs and link destinations keep
their original spelling. Pass Python f-string replacement fields as additional
protected spans when cleaning generator literals.
"""
from __future__ import annotations

import re

CYRILLIC = r"А-Яа-яЁё"
ADJACENCY = re.compile(rf"(?<=\d)(?=[{CYRILLIC}])|(?<=[{CYRILLIC}])(?=\d)")
PROTECTED = re.compile(
    r"(?m:^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\1[^\n]*(?:\n|$))"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|\$\$[\s\S]*?\$\$|\$[^$\n]+\$"
    r"|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]"
    r"|(?:https?://|mailto:)[^\s<>]+"
    r"|(?!\d)\b\w*_\w+\b"
)


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Return spans copied verbatim, including balanced Markdown destinations."""
    spans = [match.span() for match in PROTECTED.finditer(text)]
    for match in re.finditer(r"\]\(", text):
        depth, pos = 1, match.end()
        while pos < len(text) and depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        if not depth:
            spans.append((match.end() - 1, pos))
    # Reference-style link destinations and titles are code-like metadata.
    spans.extend(match.span() for match in re.finditer(r"(?m)^ {0,3}\[[^\]]+\]:[^\n]*", text))
    return sorted(spans)


def prose_segments(text: str, extra_protected: tuple[tuple[int, int], ...] = ()):
    cursor = 0
    for start, end in sorted(protected_spans(text) + list(extra_protected)):
        if start > cursor:
            yield False, text[cursor:start]
        if end > cursor:
            yield True, text[max(cursor, start):end]
        cursor = max(cursor, end)
    if cursor < len(text):
        yield False, text[cursor:]


def clean_report_prose(text: str, extra_protected: tuple[tuple[int, int], ...] = ()) -> str:
    return "".join(part if protected else ADJACENCY.sub(" ", part)
                   for protected, part in prose_segments(text, extra_protected))


def count_prose_adjacencies(text: str) -> int:
    return sum(len(ADJACENCY.findall(part)) for protected, part in prose_segments(text) if not protected)
