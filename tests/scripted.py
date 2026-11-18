"""A chat model that replies from a script, for agent tests."""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedModel(BaseChatModel):
    """Replies from a script; records what it was sent and which tools it could call."""

    script: list[Any]  # AIMessage replies, or exceptions to raise
    sent: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([t.name for t in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.sent.append(list(messages))
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def hand_in(*findings: dict[str, Any], call_id: str = "s1", count: int | None = None) -> AIMessage:
    """One turn handing in each finding with its own call, then saying how many."""
    calls: list[dict[str, Any]] = [
        {"name": "add_finding", "args": f, "id": f"{call_id}.{n}"}
        for n, f in enumerate(findings, start=1)
    ]
    calls.append(
        {
            "name": "submit_findings",
            "args": {"count": len(findings) if count is None else count},
            "id": call_id,
        }
    )
    return AIMessage(content="", tool_calls=calls)


def done(count: int, call_id: str = "d1") -> AIMessage:
    return call("submit_findings", {"count": count}, call_id)


def add(finding: dict[str, Any], call_id: str = "a1") -> AIMessage:
    return call("add_finding", finding, call_id)
