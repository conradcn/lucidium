"""Post-processing for LLM-emitted beat text.

The storyteller occasionally leaks a character id slug into the
narrative text — ``"dr-mira-sol: She steps into the lamplight..."``
— when it should have set ``speaker_id`` to that id and left the
text alone. The slug then ships to the player verbatim and reads
as a glaring engine-bug artifact.

This module strips the leading slug at post-processing time so the
shipped beat reads as plain prose. The character's identity is
preserved separately on ``DialogNode.speaker_id`` (and the engine
re-attaches the speaker's name in the renderer), so dropping the
slug from the text doesn't lose any information the player
actually needs.
"""

from __future__ import annotations

import re

# Slug pattern, anchored at the start of the text:
#
#   ^         start of text
#   ([a-z]    lowercase letter (NEVER a real sentence opener — those
#             use uppercase or quotes; restricting to lowercase
#             keeps "Note:" / "Warning:" / "He said:" prose intact);
#    [a-z0-9_]*   any number of lowercase / digit / underscore;
#    -         REQUIRE at least one dash (single-word colon-prefixed
#               sentences like "darkness: ..." are real prose and
#               must NOT be stripped — character ids are always
#               hyphenated, e.g. ``mira-quill``, ``dr-mira-sol``);
#    [a-z0-9_-]*) more of the slug body;
#   :\s+      colon + at least one whitespace char (so "code:42"
#             style numerics aren't misread as slugs).
_SLUG_PATTERN = re.compile(r"^([a-z][a-z0-9_]*-[a-z0-9_-]*):\s+")


def strip_speaker_slug(text: str) -> str:
    """Remove a leading ``character-id:`` slug from a beat's text.

    Case-insensitive on the character id is intentionally NOT
    supported — capitalised text starting with what looks like a
    slug is almost always real prose ("Trust-First: a manifesto"
    in some genre-fiction setting). Slugs are an LLM-output
    artifact and use the lowercase-with-dashes shape the engine
    asks for in ``new_characters[].id``.

    No-op when:
      * Text doesn't start with a slug-like token (the common case).
      * Text starts with a single-word colon prefix ("Note:") —
        the dash requirement excludes this.
      * Text starts with a quoted dialog line (the quote precedes
        any slug).

    Strips at most one leading slug. If for some reason the LLM
    emitted ``"dr-mira-sol: alex-stone: ..."`` only the first slug
    comes off; the second one is far less likely to be an artifact
    and might be intentional in the prose.
    """
    if not text:
        return text
    return _SLUG_PATTERN.sub("", text, count=1)


__all__ = ["strip_speaker_slug"]
