"""A deterministic chat model for tests: replays scripted AIMessages instead of calling an LLM."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    """A tool call for an AIMessage."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Replays scripted replies.

    `scripts` maps a key to the replies for one agent. With `routes` (key -> phrase), each
    call picks the script whose phrase appears in the system prompt, so one instance can
    play a coordinator and its sub-agents, including sub-agents running in parallel.
    Without routes, every call uses the "default" script.
    """

    scripts: dict[str, list[AIMessage]]
    routes: dict[str, str] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: list[BaseMessage]) -> ChatResult:
        system = next((m for m in messages if isinstance(m, SystemMessage)), None)
        prompt = system.text if system else ""
        key = next((k for k, phrase in self.routes.items() if phrase in prompt), "default")
        script = self.scripts.get(key)
        if not script:
            raise AssertionError(f"Script {key!r} has no replies left")
        self.calls.append(key)
        return ChatResult(generations=[ChatGeneration(message=script.pop(0))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._next(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._next(messages)

    def bind_tools(self, tools, **kwargs):
        return self
