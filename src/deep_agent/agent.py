"""The case-resolution deep agent.

A planning coordinator delegates to three read-only investigator sub-agents. Each
sub-agent sees only the tools its job needs, works in its own context window, and writes
its evidence to the case's virtual filesystem. Company policies are mounted read-only at
/policies/. Case files live in graph state, so the checkpointer persists them per case.
"""

from __future__ import annotations

from pathlib import Path

from deepagents import (
    FilesystemPermission,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    SubAgent,
    create_deep_agent,
    register_harness_profile,
)
from deepagents._models import get_model_identifier, get_model_provider
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from src.deep_agent.actions import (
    EXECUTE_REFUND,
    CaseStateMiddleware,
    build_action_tools,
    refund_review_card,
)
from src.deep_agent.backend import CloudOrderClient
from src.deep_agent.config import DeepAgentSettings, get_settings
from src.deep_agent.context import CaseContext
from src.deep_agent.prompts import (
    BILLING_PROMPT,
    COORDINATOR_PROMPT,
    FULFILLMENT_PROMPT,
    POLICY_PROMPT,
    SUBAGENT_DESCRIPTIONS,
)
from src.deep_agent.tools import build_read_tools

POLICIES_DIR = Path(__file__).parent / "policies"

COORDINATOR_TOOLS = ("get_customer_overview", "get_order", "get_support_tickets",
                     "propose_refund", EXECUTE_REFUND, "escalate_case")

# The only gate on money movement. Reviewers may approve, lower the amount, or reject.
APPROVAL_GATES = {
    EXECUTE_REFUND: {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": refund_review_card,
    },
}
SUBAGENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "billing-investigator": (BILLING_PROMPT,
                             ("get_order", "get_payment_ledger", "get_refund_history")),
    "fulfillment-investigator": (FULFILLMENT_PROMPT, ("get_order", "get_shipment_tracking")),
    "policy-analyst": (POLICY_PROMPT, ("get_order", "check_refund_eligibility")),
}

# Policies are company documents: every agent may read them, none may change them.
POLICY_PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=["/policies/**"], mode="deny"),
]


def case_backend() -> CompositeBackend:
    """Case files in graph state (checkpointed per case); policies from disk, read-only."""
    return CompositeBackend(
        default=StateBackend(),
        routes={"/policies/": FilesystemBackend(root_dir=POLICIES_DIR, virtual_mode=True)},
    )


def _disable_general_purpose_subagent(model: BaseChatModel) -> None:
    """Turn off deepagents' auto-added `general-purpose` sub-agent for this model.

    That sub-agent inherits every coordinator tool, which would let a delegated task use
    tools outside the three investigators' read-only sets. deepagents 0.7 only exposes this
    switch through a harness profile, so register one keyed to this exact model.
    """
    provider = get_model_provider(model)
    identifier = get_model_identifier(model)
    key = f"{provider}:{identifier}" if identifier and ":" not in identifier else provider
    if not key:
        raise RuntimeError(f"Cannot resolve a harness-profile key for {type(model).__name__}")
    register_harness_profile(
        key, HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))
    )


def build_case_agent(
    *,
    model: BaseChatModel,
    subagent_model: BaseChatModel,
    client: CloudOrderClient,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile the coordinator graph. Run it with `context=CaseContext(...)`.

    Sub-agents get read tools only. The action tools exist only on the coordinator, and
    execute_refund always pauses for human approval, so the checkpointer is required.
    """
    tools = {**build_read_tools(client), **build_action_tools(client)}
    _disable_general_purpose_subagent(model)
    subagents: list[SubAgent] = [
        {
            "name": name,
            "description": SUBAGENT_DESCRIPTIONS[name],
            "system_prompt": prompt,
            "tools": [tools[tool_name] for tool_name in tool_names],
            "model": subagent_model,
            "interrupt_on": {},  # read-only: nothing to approve
        }
        for name, (prompt, tool_names) in SUBAGENTS.items()
    ]
    return create_deep_agent(
        model=model,
        tools=[tools[name] for name in COORDINATOR_TOOLS],
        system_prompt=COORDINATOR_PROMPT,
        middleware=[TodoListMiddleware(), CaseStateMiddleware()],
        subagents=subagents,
        backend=case_backend(),
        permissions=POLICY_PERMISSIONS,
        interrupt_on=APPROVAL_GATES,
        context_schema=CaseContext,
        checkpointer=checkpointer or InMemorySaver(),
        name="case-coordinator",
    )


def create_models(settings: DeepAgentSettings) -> tuple[BaseChatModel, BaseChatModel]:
    if settings.anthropic_api_key is None:
        raise RuntimeError("Set ANTHROPIC_API_KEY (or DEEP_AGENT_ANTHROPIC_API_KEY) to run the agent.")
    key = settings.anthropic_api_key.get_secret_value()
    coordinator = ChatAnthropic(model=settings.coordinator_model, api_key=key, max_tokens=4096)
    subagent = ChatAnthropic(model=settings.subagent_model, api_key=key, max_tokens=2048)
    return coordinator, subagent


def build_case_agent_from_settings(
    client: CloudOrderClient | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    settings: DeepAgentSettings | None = None,
):
    settings = settings or get_settings()
    coordinator, subagent = create_models(settings)
    return build_case_agent(
        model=coordinator,
        subagent_model=subagent,
        client=client or CloudOrderClient.from_settings(settings),
        checkpointer=checkpointer,
    )
