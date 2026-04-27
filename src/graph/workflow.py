from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
from src.router.intent import classify_intent
from src.agents.faq_agent import run_faq_agent
from src.agents.order_agent import run_order_agent
from src.agents.cs_agent import run_cs_agent
from src.memory.store import load_history, save_history


class AgentState(TypedDict):
    messages: list
    intent: str
    session_id: str
    user_id: str | None
    response: str


@traceable(name="router_node")
async def router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    history = state["messages"][:-1] if len(state["messages"]) > 1 else None
    intent = await classify_intent(content, history)
    return {"intent": intent}


@traceable(name="faq_node")
async def faq_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    response = await run_faq_agent(query)
    return {"response": response}


@traceable(name="order_node")
async def order_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    response = await run_order_agent(query, state.get("user_id"))
    return {"response": response}


@traceable(name="cs_react_node")
async def cs_react_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    response = await run_cs_agent(query, state.get("user_id"))
    return {"response": response}


def route_by_intent(state: AgentState) -> Literal["faq_node", "order_node", "cs_react_node"]:
    intent = state.get("intent", "complex")
    if intent == "faq":
        return "faq_node"
    elif intent == "order":
        return "order_node"
    else:
        return "cs_react_node"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router_node", router_node)
    graph.add_node("faq_node", faq_node)
    graph.add_node("order_node", order_node)
    graph.add_node("cs_react_node", cs_react_node)

    graph.add_edge(START, "router_node")
    graph.add_conditional_edges("router_node", route_by_intent)
    graph.add_edge("faq_node", END)
    graph.add_edge("order_node", END)
    graph.add_edge("cs_react_node", END)

    return graph.compile()


workflow = build_graph()


async def run_workflow(
    message: str, session_id: str, user_id: str | None = None
) -> tuple[str, str]:
    history = await load_history(session_id)
    history.append(HumanMessage(content=message))

    result = await workflow.ainvoke(
        {
            "messages": history,
            "intent": "",
            "session_id": session_id,
            "user_id": user_id,
            "response": "",
        },
        config={
            "metadata": {
                "session_id": session_id,
                "user_id": user_id or "anonymous",
            }
        },
    )

    response_text = result.get("response", "I'm sorry, I wasn't able to process your request.")
    intent = result.get("intent", "unknown")

    history.append(AIMessage(content=response_text))
    await save_history(session_id, history)

    return response_text, intent
