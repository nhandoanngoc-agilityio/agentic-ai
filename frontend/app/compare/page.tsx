"use client";

import { useState } from "react";
import { useAgent } from "@copilotkit/react-core/v2";
import { AgentPanel } from "@/components/AgentPanel";
import { ResultDetails, type RunStats } from "@/components/ResultDetails";
import type { AgentStateMessage, ReportInterruptPayload } from "@/lib/types";

interface PendingInterrupt {
  payload: ReportInterruptPayload;
  resolve: (result: string) => void;
  draftReadyAtMs: number;
}

type Provider = "openai" | "anthropic";

const AGENT_ID_BY_PROVIDER: Record<Provider, string> = {
  openai: "openaiAgent",
  anthropic: "anthropicAgent",
};

const LABEL_BY_PROVIDER: Record<Provider, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
};

export default function ComparePage() {
  const [provider, setProvider] = useState<Provider>("openai");
  const [objective, setObjective] = useState("");
  const [running, setRunning] = useState(false);
  const [startedAtMs, setStartedAtMs] = useState(0);
  const [pending, setPending] = useState<PendingInterrupt | null>(null);

  const agentId = AGENT_ID_BY_PROVIDER[provider];
  const { agent } = useAgent({ agentId });

  function handleRun() {
    setPending(null);
    // Updater form so the impure `Date.now()` reading happens when React applies the
    // update rather than inline in a function `react-hooks/purity` can't prove is only
    // ever called from an event handler (matches the pattern already used elsewhere).
    setStartedAtMs(() => Date.now());
    setRunning(true);
    agent.setState({
      messages: [],
      objective,
      next: "research",
      research_findings: [],
      analytics_results: [],
      report_path: null,
    });
    agent.runAgent();
  }

  function handleInterrupt(payload: ReportInterruptPayload, resolve: (r: string) => void) {
    setPending({ payload, resolve, draftReadyAtMs: Date.now() });
  }

  // Interrupt/resume asymmetry, both halves: the interrupt payload *arrives* as a JSON
  // string that AgentPanel.tsx has to parse (see its comment), and a resume has to be
  // *sent* back as a JSON string too -- hence `JSON.stringify` on every `resolve()` call
  // below. Don't "simplify" either side to match the other; they're stringly-typed on
  // purpose because that's what CopilotKit's legacy `on_interrupt` path expects.
  function handleApprove() {
    pending?.resolve(JSON.stringify({ approved: true }));
    setPending(null);
    setRunning(false);
  }

  function handleReject(feedback?: string) {
    pending?.resolve(JSON.stringify({ approved: false, feedback }));
    setPending(null);
    // Stays `running`: a rejection triggers a redraft, which fires another interrupt.
  }

  // Dead end: the agent stopped without ever leaving a draft to review, and there's no
  // pending interrupt left to resolve -- nothing else would ever clear `running`.
  const runFailed = running && !agent.isRunning && pending === null && Boolean(agent.state?.error);

  // Unconditional escape hatch. `runFailed` only covers *in-graph* failures (guardrails.py
  // is what writes `state.error`); if the browser never reaches the selected `langgraph dev`
  // deployment at all -- wrong port, process not started -- `isRunning` goes false and no
  // error is ever set, so nothing would clear `running`. Rather than trying to tell that
  // apart from a run that's legitimately still going, always offer a manual Reset.
  function handleReset() {
    setPending(null);
    setRunning(false);
  }

  const stats: RunStats | null = pending
    ? {
        analyticsResults: (agent.state?.analytics_results ?? []).length,
        supervisorVisits: (agent.state?.messages ?? []).filter(
          (m: AgentStateMessage) => m.name === "supervisor",
        ).length,
        elapsedMs: pending.draftReadyAtMs - startedAtMs,
      }
    : null;

  return (
    <main>
      <div style={{ padding: "1.5rem" }}>
        <label htmlFor="provider">Provider</label>
        <select
          id="provider"
          value={provider}
          disabled={running && !runFailed}
          onChange={(e) => {
            setProvider(e.target.value as Provider);
            // The select is only interactive when a run isn't legitimately in
            // progress (see `disabled` above): either no run at all (this is a
            // no-op) or `runFailed` (this is the reset that unblocks retry with
            // the newly selected provider -- otherwise `running` stays stuck at
            // `true` from the failed run and both the select and Run button
            // re-disable for the new provider too).
            setRunning(false);
          }}
          style={{ marginLeft: "0.5rem" }}
        >
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
        </select>

        <div style={{ marginTop: "0.5rem" }}>
          <label htmlFor="objective">Research objective</label>
          <input
            id="objective"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            style={{ width: "100%", marginTop: "0.5rem" }}
          />
        </div>
        <button
          type="button"
          onClick={handleRun}
          disabled={!objective || (running && !runFailed)}
          style={{ marginTop: "0.5rem" }}
        >
          Run
        </button>
        {running && (
          <button
            type="button"
            onClick={handleReset}
            style={{ marginTop: "0.5rem", marginLeft: "0.5rem" }}
          >
            Reset
          </button>
        )}
        {runFailed && (
          <p className="agent-panel-error">Run ended without a draft ({agent.state?.error}). Try again.</p>
        )}
      </div>

      <AgentPanel agentId={agentId} label={LABEL_BY_PROVIDER[provider]} onInterrupt={handleInterrupt} />

      {pending && stats && (
        <div style={{ padding: "1.5rem" }}>
          <ResultDetails
            payload={pending.payload}
            findings={agent.state?.research_findings ?? []}
            results={agent.state?.analytics_results ?? []}
            stats={stats}
            onApprove={handleApprove}
            onReject={handleReject}
          />
        </div>
      )}
    </main>
  );
}
