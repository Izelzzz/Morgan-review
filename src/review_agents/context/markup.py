"""Marks repository and pull request content as data, never as instructions."""

import html
import re

_CLOSING_TAG = re.compile(r"</\s*untrusted", re.IGNORECASE)


def untrusted(source: str, text: str) -> str:
    """Wrap `text` in a data block; a closing tag inside it is defused, so it cannot end it."""
    safe = _CLOSING_TAG.sub("&lt;/untrusted", text)
    return f'<untrusted source="{html.escape(source, quote=True)}">\n{safe}\n</untrusted>'
