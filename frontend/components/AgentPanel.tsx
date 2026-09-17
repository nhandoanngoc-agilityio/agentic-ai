"use client";

import { useAgent, useInterrupt } from "@copilotkit/react-core/v2";
import type { ReportInterruptPayload } from "@/lib/types";

export interface AgentPanelProps {
  agentId: string;
  label: string;
  onInterrupt: (payload: ReportInterruptPayload, resolve: (result: string) => void) => void;
}

export function AgentPanel({ agentId, label, onInterrupt }: AgentPanelProps) {
  const { agent } = useAgent({ agentId });

  useInterrupt({
    agentId,
    renderInChat: false,
    // `handler`, not `render`: CopilotKit calls `render` during React's render
    // phase (inside a useMemo), so notifying the parent from there updates
    // another component mid-render. `handler` runs from a useEffect keyed on
    // the pending interrupt, i.e. after commit.
    handler: ({ event, resolve }) => {
      // CopilotKit's legacy `on_interrupt` path (the one in use here -- route.ts
      // leaves `emitInterruptOutcome` off) JSON-stringifies the interrupt value
      // before emitting it, so `event.value` arrives as a string, not an object.
      // The object branch is a guard in case a future version/code path delivers
      // it already parsed.
      const raw: unknown = event.value;
      const payload = (typeof raw === "string" ? JSON.parse(raw) : raw) as ReportInterruptPayload;
      onInterrupt(payload, resolve);
    },
    render: () => <></>,
  });

  const state = agent.state ?? {};
  const messages = state.messages ?? [];
  const lastNode = messages.length > 0 ? messages[messages.length - 1]?.name ?? "supervisor" : null;
  const findingsCount = (state.research_findings ?? []).length;
  const resultsCount = (state.analytics_results ?? []).length;
  const errorText = state.error;

  return (
    <div className="agent-panel">
      <h2>{label}</h2>
      <p>Status: {agent.isRunning ? (lastNode ?? "starting") : lastNode ? "paused" : "idle"}</p>
      <p>Research findings: {findingsCount}</p>
      <p>Analytics results: {resultsCount}</p>
      {errorText && <p className="agent-panel-error">Error: {errorText}</p>}
    </div>
  );
}
