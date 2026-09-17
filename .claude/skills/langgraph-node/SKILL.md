---
name: langgraph-node
description: Use when adding or changing a graph node, sub-agent, or supervisor route in this LangGraph project. Covers state fields, error-boundary registration, routing, and the hermetic test pattern.
---

# Adding or changing a graph node

1. **State first.** Add any new fields to `AgentState` in `src/market_research_team/state.py`. Use `NotRequired[...]` for optional fields. If a new route is needed, extend `RouteDecision`.
2. **Node module.** Create `src/market_research_team/agents/<name>/node.py` with
   `def <name>_node(state: AgentState) -> dict[str, Any]` returning only the keys it changes. Keep LLM/tool calls behind a module-level function (e.g. `_run_pipeline`) so tests can monkeypatch it.
3. **Register with the boundary.** In `graph.py`:
   `builder.add_node("<name>", with_error_boundary("<name>", <name>_node))`
   then `builder.add_edge("<name>", "supervisor")`.
4. **Route.** Add `"<name>": "<name>"` to the `add_conditional_edges` map in `graph.py`, and teach `decide_next_step` in `agents/supervisor/router.py` when to pick it (update the routing prompt and `_SupervisorDecision`).
5. **Test hermetically.** Copy the pattern in `tests/agents/test_supervisor_routing.py`: monkeypatch the node module's pipeline function with a fake, build the graph with `build_production_graph(InMemorySaver())`, invoke with a `thread_id`, and assert on the returned state. Also add a node-level test like those in `tests/agents/test_reporting_node.py`. No real LLM calls.
6. **Run** the `test-runner` agent, then `ruff check src tests`.
7. Update the architecture diagram in `README.md` and `docs/architecture.md` if the topology changed.
