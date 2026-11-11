"""Keeping an agent's tool conversation inside the model's window (decision D-030).

Two layers, neither calls a model or loses code for good: a tool result over its limit
keeps its head and tail and says how to read the middle; and when the conversation
would no longer fit the window, the oldest results are cut to a short head and tail
first. A third guard notices an agent repeating the exact same call.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from review_agents.context.markup import untrusted
from review_agents.context.tokens import estimate_tokens

# Roughly three bytes to a token (decision D-019), for a line too long to keep whole.
_BYTES_PER_TOKEN = 3
REPEATED = "You have made this exact call"
REMOVED = "[Removed to fit the model's window: call the tool again if you still need it.]"
# The same call (tool and arguments) this many times gets a reminder (decision D-030).
REPEAT_REMINDER_AT = 3


class ConversationTooLong(RuntimeError):
    """Even with every tool result removed, the conversation does not fit the window."""


@dataclass(frozen=True)
class ConversationLimits:
    """Server settings; the defaults are the decided values (D-030), windowless."""

    window: int = 10**9  # the model's context window, in tokens
    result_tokens: int = 10_000  # one tool result at most
    shortened_tokens: int = 500  # head and tail each, once an older result gives way
    output_reserve: int = 4_000  # kept free for the model's reply


def _take(lines: list[str], tokens: int) -> list[str]:
    """Whole lines from the start of `lines` within `tokens`."""
    taken: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > tokens:
            break
        taken.append(line)
        used += cost
    return taken


def shorten(text: str, keep_tokens: int, tool: str) -> str:
    """`text`, or its head and tail of about `keep_tokens` each with a note on the rest."""
    total = estimate_tokens(text)
    if total <= 2 * keep_tokens:
        return text
    how = (
        "The kept lines show their numbers; read_file with start_line and end_line reads the rest."
        if tool == "read_file"
        else "Narrow the request to see the rest."
    )
    lines = text.splitlines()
    head = _take(lines, keep_tokens)
    tail = list(reversed(_take(list(reversed(lines)), keep_tokens)))
    left_out = len(lines) - len(head) - len(tail)
    if head and tail and left_out > 0:
        note = (
            f"[Shortened: this result was about {total} tokens in {len(lines)} lines;"
            f" {left_out} lines in the middle are left out. {how}]"
        )
        return "\n".join([note, *head, f"[... {left_out} lines left out ...]", *tail])
    # Lines too long to keep whole, such as minified code: cut by characters instead.
    chars = keep_tokens * _BYTES_PER_TOKEN
    while chars > 1 and max(estimate_tokens(text[:chars]), estimate_tokens(text[-chars:])) > (
        keep_tokens
    ):
        chars //= 2  # a script of more than a token a byte, such as Chinese
    note = (
        f"[Shortened: this result was about {total} tokens; the middle"
        f" {len(text) - 2 * chars} characters are left out. {how}]"
    )
    return "\n".join([note, text[:chars], "[... left out ...]", text[-chars:]])


def conversation_tokens(messages: Sequence[BaseMessage]) -> int:
    """About how many tokens `messages` take: their text and their tool calls."""
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.content))
        if isinstance(message, AIMessage) and message.tool_calls:
            total += estimate_tokens(json.dumps(message.tool_calls, default=str))
    return total


@dataclass
class _Result:
    index: int  # where its ToolMessage sits in the conversation
    tool: str
    key: str  # the call, tool and arguments, as counted for repeats
    text: str  # the result as the tool returned it, before any wrapping
    suffix: str  # what follows the data block, such as a repeat reminder
    shortened: bool = False
    removed: bool = False


@dataclass
class Conversation:
    """The tool results of one agent's conversation, so they can be cut and re-wrapped."""

    limits: ConversationLimits
    # Tokens every request carries besides the messages: the tool definitions.
    overhead: int = 0
    _results: list[_Result] = field(default_factory=list)
    _calls: Counter[str] = field(default_factory=Counter)

    def answer(
        self, messages: list[BaseMessage], call_id: str | None, tool: str, args: Any, result: str
    ) -> None:
        """Append a tool's result: within its limit, in its data block, reminded if repeated."""
        key = json.dumps([tool, args], sort_keys=True, default=str)
        self._calls[key] += 1
        times = self._calls[key]
        suffix = (
            f"\n\n{REPEATED} {times} times. Change your approach, or submit your findings."
            if times >= REPEAT_REMINDER_AT
            else ""
        )
        text = shorten(result, self.limits.result_tokens // 2, tool)
        self._results.append(_Result(len(messages), tool, key, text, suffix))
        messages.append(_message(tool, text, suffix, call_id))

    def fit(self, messages: list[BaseMessage]) -> None:
        """Make the conversation and a reply fit the window, or raise ConversationTooLong.

        The oldest results are cut to a short head and tail first; if that is not enough,
        the oldest are removed, each leaving a line saying it can be read again. Only
        when the rest alone is too large does this give up: a request larger than the
        window is never sent.
        """
        room = self.limits.window - self.limits.output_reserve - self.overhead
        total = conversation_tokens(messages)  # counted once; each change adjusts it
        for removing in (False, True):
            for result in self._results:
                if total <= room:
                    return
                if result.removed or (result.shortened and not removing):
                    continue
                old = messages[result.index]
                assert isinstance(old, ToolMessage)  # noqa: S101 - set by answer()
                if removing:
                    result.removed = True
                    # It was removed, not seen: calling it again is not repeating oneself.
                    self._calls[result.key] -= 1
                    new = ToolMessage(REMOVED, tool_call_id=old.tool_call_id)
                else:
                    result.text = shorten(result.text, self.limits.shortened_tokens, result.tool)
                    result.shortened = True
                    new = _message(result.tool, result.text, result.suffix, old.tool_call_id)
                messages[result.index] = new
                total += estimate_tokens(str(new.content)) - estimate_tokens(str(old.content))
        if total > room:
            raise ConversationTooLong(
                f"the conversation needs about {total} tokens, more than the {room} the"
                " model's window leaves after the tool definitions and the room kept for"
                " the reply"
            )


def _message(tool: str, text: str, suffix: str, call_id: str | None) -> ToolMessage:
    return ToolMessage(untrusted(f"tool:{tool}", text) + suffix, tool_call_id=call_id)
