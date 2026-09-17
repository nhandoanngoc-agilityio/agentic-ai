import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AgentPanel } from "./AgentPanel";
import type { ReportInterruptPayload } from "@/lib/types";

const mockUseAgent = vi.fn();
const mockUseInterrupt = vi.fn();

vi.mock("@copilotkit/react-core/v2", () => ({
  useAgent: (args: unknown) => mockUseAgent(args),
  useInterrupt: (args: unknown) => mockUseInterrupt(args),
}));

type InterruptCallback = (args: {
  event: { value: unknown };
  resolve: (s: string) => void;
}) => unknown;

/** Capture the `handler` / `render` AgentPanel hands to useInterrupt, so a test can
 *  invoke either one directly and see which of them notifies the parent. */
function captureInterruptConfig() {
  let config: { handler?: InterruptCallback; render?: InterruptCallback } | undefined;
  mockUseInterrupt.mockImplementation((args: typeof config) => {
    config = args;
    return null;
  });
  return {
    capturedHandler: () => config?.handler,
    capturedRender: () => config?.render,
  };
}

describe("AgentPanel", () => {
  it("renders progress counts from agent state", () => {
    mockUseAgent.mockReturnValue({
      agent: {
        isRunning: true,
        state: {
          messages: [{ name: "analytics_agent", content: "..." }],
          research_findings: [{ source: "a", content: "b", relevance_score: 0.5 }],
          analytics_results: [],
          error: null,
        },
      },
    });
    mockUseInterrupt.mockReturnValue(null);

    render(<AgentPanel agentId="anthropicAgent" label="Anthropic" onInterrupt={vi.fn()} />);

    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText(/analytics_agent/)).toBeInTheDocument();
    expect(screen.getByText(/Research findings: 1/)).toBeInTheDocument();
  });

  it("renders an error when agent state carries one", () => {
    mockUseAgent.mockReturnValue({
      agent: {
        isRunning: false,
        state: {
          messages: [],
          research_findings: [],
          analytics_results: [],
          error: "Recursion limit reached",
        },
      },
    });
    mockUseInterrupt.mockReturnValue(null);

    render(<AgentPanel agentId="openaiAgent" label="OpenAI" onInterrupt={vi.fn()} />);

    expect(screen.getByText(/Recursion limit reached/)).toBeInTheDocument();
  });

  it("parses the JSON-string interrupt value CopilotKit delivers and forwards it", () => {
    mockUseAgent.mockReturnValue({
      agent: { isRunning: true, state: { messages: [], research_findings: [], analytics_results: [] } },
    });
    const onInterrupt = vi.fn();
    const { capturedHandler } = captureInterruptConfig();

    render(<AgentPanel agentId="anthropicAgent" label="Anthropic" onInterrupt={onInterrupt} />);

    const payload: ReportInterruptPayload = {
      action: "write_report",
      filename: "x.md",
      content: "# Draft",
      attempt: 1,
      max_attempts: 3,
    };
    const resolve = vi.fn();
    // CopilotKit's legacy on_interrupt path emits `JSON.stringify(interrupt.value)`,
    // so the value arrives as a string -- not the already-parsed object.
    capturedHandler()?.({ event: { value: JSON.stringify(payload) }, resolve });

    expect(onInterrupt).toHaveBeenCalledWith(payload, resolve);
    expect(onInterrupt.mock.calls[0][0].content).toBe("# Draft");
  });

  it("also accepts an already-parsed interrupt value", () => {
    mockUseAgent.mockReturnValue({
      agent: { isRunning: true, state: { messages: [], research_findings: [], analytics_results: [] } },
    });
    const onInterrupt = vi.fn();
    const { capturedHandler } = captureInterruptConfig();

    render(<AgentPanel agentId="openaiAgent" label="OpenAI" onInterrupt={onInterrupt} />);

    const payload: ReportInterruptPayload = {
      action: "write_report",
      filename: "y.md",
      content: "# Other draft",
      attempt: 2,
      max_attempts: 3,
    };
    const resolve = vi.fn();
    capturedHandler()?.({ event: { value: payload }, resolve });

    expect(onInterrupt).toHaveBeenCalledWith(payload, resolve);
  });

  it("notifies the parent from `handler`, not from `render`", () => {
    mockUseAgent.mockReturnValue({
      agent: { isRunning: true, state: { messages: [], research_findings: [], analytics_results: [] } },
    });
    const onInterrupt = vi.fn();
    const { capturedRender } = captureInterruptConfig();

    render(<AgentPanel agentId="anthropicAgent" label="Anthropic" onInterrupt={onInterrupt} />);

    // CopilotKit calls `render` during React's render phase (inside a useMemo), so
    // it must stay side-effect free -- updating the parent from there triggers
    // React's "cannot update a component while rendering a different one" error.
    capturedRender()?.({
      event: { value: JSON.stringify({ action: "write_report", content: "# Draft" }) },
      resolve: vi.fn(),
    });

    expect(onInterrupt).not.toHaveBeenCalled();
  });
});
