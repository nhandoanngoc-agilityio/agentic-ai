"""Top-level graph assembly.
"""

from langgraph.graph import END, START, MessagesState, StateGraph


def _placeholder_node(state: MessagesState) -> MessagesState:
    return state


builder = StateGraph(MessagesState)
builder.add_node("placeholder", _placeholder_node)
builder.add_edge(START, "placeholder")
builder.add_edge("placeholder", END)

graph = builder.compile()
