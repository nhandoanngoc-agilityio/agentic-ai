"use client";

import { useState } from "react";
import type { AnalyticsResult, ReportInterruptPayload, ResearchFinding } from "@/lib/types";

export interface RunStats {
  /** How many analytics metrics the run computed -- NOT a tool-call count. The frontend
   *  never sees tool calls; label this for what it measures, not what it sounds like. */
  analyticsResults: number;
  /** How many times the supervisor was visited. The supervisor routes before *every*
   *  node, so this is total routing hops, not research passes specifically. */
  supervisorVisits: number;
  elapsedMs: number;
}

export interface ResultDetailsProps {
  payload: ReportInterruptPayload;
  findings: ResearchFinding[];
  results: AnalyticsResult[];
  stats: RunStats;
  onApprove: () => void;
  onReject: (feedback?: string) => void;
}

export function ResultDetails({
  payload,
  findings,
  results,
  stats,
  onApprove,
  onReject,
}: ResultDetailsProps) {
  const [feedback, setFeedback] = useState("");

  return (
    <div className="agent-panel">
      <h3>
        Draft (attempt {payload.attempt} of {payload.max_attempts})
      </h3>
      <pre style={{ whiteSpace: "pre-wrap" }}>{payload.content}</pre>

      <h3>Research findings</h3>
      {findings.length === 0 ? (
        <p>No research findings.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Content</th>
              <th>Relevance</th>
            </tr>
          </thead>
          <tbody>
            {findings.map((finding, index) => (
              <tr key={index}>
                <td>{finding.source}</td>
                <td>{finding.content}</td>
                <td>{finding.relevance_score}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Analytics results</h3>
      {results.length === 0 ? (
        <p>No analytics results.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result, index) => (
              <tr key={index}>
                <td>{result.metric}</td>
                <td>{result.value}</td>
                <td>{result.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p>
        Research findings: {findings.length} · Analytics results: {stats.analyticsResults} ·
        Supervisor visits: {stats.supervisorVisits} · Elapsed:{" "}
        {Math.round(stats.elapsedMs / 1000)}s
      </p>

      <button type="button" onClick={onApprove}>
        Approve
      </button>
      <textarea
        aria-label="Rejection feedback"
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        placeholder="Optional feedback for the next draft"
      />
      <button type="button" onClick={() => onReject(feedback || undefined)}>
        Reject with feedback
      </button>
    </div>
  );
}
