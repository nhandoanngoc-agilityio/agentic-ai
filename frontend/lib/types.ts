export interface ResearchFinding {
  source: string;
  content: string;
  relevance_score: number;
}

export interface AnalyticsResult {
  metric: string;
  value: number;
  detail: string;
}

export interface AgentStateMessage {
  name?: string | null;
  content: string;
}

/** Mirrors src/market_research_team/state.py's AgentState. */
export interface ComparisonAgentState {
  messages?: AgentStateMessage[];
  objective?: string;
  next?: "research" | "analytics" | "reporting" | "FINISH";
  research_findings?: ResearchFinding[];
  analytics_results?: AnalyticsResult[];
  report_path?: string | null;
  error?: string | null;
  /** Set by reporting_node when a resume decision carries `discard: true`. Mirrored here
   *  to keep this type honest with `AgentState`; this frontend never sends `discard`
   *  (single-provider mode only ever approves or rejects-with-feedback), so the field
   *  never arrives set -- the backend path is retained unused on purpose. */
  report_discarded?: boolean;
}

/** Mirrors the interrupt() payload from reporting_node (node.py). */
export interface ReportInterruptPayload {
  action: "write_report";
  filename: string;
  content: string;
  attempt: number;
  max_attempts: number;
}

/** Mirrors the resume-decision shape reporting_node reads (node.py, Task 1 of this plan). */
export interface ResumeDecision {
  approved: boolean;
  feedback?: string | null;
  discard?: boolean;
}
