"""Token counts for sizing context; never used for billing.

Counted with tiktoken's o200k_base encoding (decision D-013). Other models' tokenizers
differ by a few percent, so callers still leave headroom. The encoding is baked into
the container image; elsewhere tiktoken downloads it once into TIKTOKEN_CACHE_DIR.
"""

import logging
import threading
import time

import tiktoken

log = logging.getLogger(__name__)

ENCODING = "o200k_base"
# Without the encoding, sizing falls back to a character estimate and tries again later.
RETRY_SECONDS = 600.0


def _load_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


class _Loader:
    """The encoding once it loads; None, and a retry after a while, while it does not."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cache_clear()

    def cache_clear(self) -> None:
        self._encoding: tiktoken.Encoding | None = None
        self._next_try = float("-inf")

    def __call__(self) -> tiktoken.Encoding | None:
        if self._encoding is not None:  # loaded: no lock needed to read it
            return self._encoding
        with self._lock:
            if self._encoding is None and time.monotonic() >= self._next_try:
                try:
                    self._encoding = _load_encoding()
                except Exception:
                    self._next_try = time.monotonic() + RETRY_SECONDS
                    log.warning(
                        "tiktoken encoding %s could not be loaded; estimating from characters",
                        ENCODING,
                        exc_info=True,
                    )
            return self._encoding


_encoding = _Loader()


def _character_estimate(text: str) -> int:
    # About four characters per token for ASCII; CJK and other wide characters are
    # roughly one token each.
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return (ascii_chars + 3) // 4 + (len(text) - ascii_chars)


def estimate_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return _character_estimate(text)
    # PR text may contain special-token markers; they are counted as ordinary text.
    return len(encoding.encode(text, disallowed_special=()))
