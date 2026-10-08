"""Fence untrusted text (retrieved chunks, computed analytics) inside a prompt.

Each agent prompt that embeds retrieved content wraps it in a named tag and
tells the model to treat the tag's contents as data, never as instructions.
Shared so the analytics and reporting prompts fence the same way.
"""

FINDINGS_TAG = "retrieved_research_data"
ANALYTICS_TAG = "computed_analytics"
_TAGS = (FINDINGS_TAG, ANALYTICS_TAG)


def escape_closing_tags(text: str) -> str:
    """Neutralize literal closing-delimiter tags in untrusted text.

    A poisoned corpus chunk containing the literal string
    `</retrieved_research_data>` would otherwise close its enclosing block
    early, letting the rest of the chunk land outside the tag as if it were
    operator text. This is delimiter escaping only (backslash-escaping the
    closing bracket), not keyword/content scanning for injection phrases.
    """

    for tag in _TAGS:
        text = text.replace(f"</{tag}>", f"<\\/{tag}>")
    return text


def fence(tag: str, text: str) -> str:
    """`text` inside `<tag>…</tag>`, with any closing tag inside it escaped."""

    return f"<{tag}>\n{escape_closing_tags(text)}\n</{tag}>"
