import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import ComparePage from "./page";

const mockUseAgent = vi.fn();
const mockUseInterrupt = vi.fn();
let interruptHandler:
  | ((args: { event: { value: unknown }; resolve: (s: string) => void }) => unknown)
  | undefined;

vi.mock("@copilotkit/react-core/v2", () => ({
  useAgent: (args: { agentId: string }) => mockUseAgent(args),
  // AgentPanel (rendered for real by page.tsx) owns the useInterrupt subscription and
  // notifies the parent from `handler` (post-commit), not `render` -- see AgentPanel.tsx.
  useInterrupt: (args: {
    handler: (r: { event: { value: unknown }; resolve: (s: string) => void }) => unknown;
  }) => {
    interruptHandler = args.handler;
    return mockUseInterrupt(args);
  },
}));

const resultDetailsProps: { current: Record<string, unknown> | null } = { current: null };

vi.mock("@/components/ResultDetails", () => ({
  ResultDetails: (props: Record<string, unknown>) => {
    resultDetailsProps.current = props;
    return (
      <div>
        <button onClick={() => (props.onApprove as () => void)()}>mock-approve</button>
        <button onClick={() => (props.onReject as (f?: string) => void)("more detail")}>
          mock-reject
        </button>
      </div>
    );
  },
}));

function fakeAgent(overrides: Record<string, unknown> = {}) {
  return {
    isRunning: true,
    state: { messages: [], research_findings: [], analytics_results: [] },
    addMessage: vi.fn(),
    setState: vi.fn(),
    runAgent: vi.fn(),
    ...overrides,
  };
}

describe("ComparePage", () => {
  it("defaults the provider dropdown to OpenAI", () => {
    mockUseAgent.mockReturnValue({ agent: fakeAgent() });
    render(<ComparePage />);

    expect(screen.getByLabelText("Provider")).toHaveValue("openai");
  });

  it("runs the OpenAI agent by default", () => {
    const openaiAgent = fakeAgent();
    mockUseAgent.mockImplementation((args: { agentId: string }) =>
      args.agentId === "openaiAgent" ? { agent: openaiAgent } : { agent: fakeAgent() },
    );
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    expect(openaiAgent.setState).toHaveBeenCalledWith(
      expect.objectContaining({ objective: "Assess pricing" }),
    );
    expect(openaiAgent.runAgent).toHaveBeenCalledTimes(1);
  });

  it("runs the Anthropic agent when selected before running", () => {
    const anthropicAgent = fakeAgent();
    mockUseAgent.mockImplementation((args: { agentId: string }) =>
      args.agentId === "anthropicAgent" ? { agent: anthropicAgent } : { agent: fakeAgent() },
    );
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "anthropic" } });
    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    expect(anthropicAgent.runAgent).toHaveBeenCalledTimes(1);
  });

  it("shows ResultDetails once the interrupt fires and resolves approval", () => {
    mockUseAgent.mockReturnValue({ agent: fakeAgent() });
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    const resolve = vi.fn();
    const payload = {
      action: "write_report",
      filename: "x.md",
      content: "# Draft",
      attempt: 1,
      max_attempts: 3,
    };
    act(() => {
      interruptHandler?.({ event: { value: JSON.stringify(payload) }, resolve });
    });

    expect(resultDetailsProps.current?.payload).toEqual(payload);
    fireEvent.click(screen.getByText("mock-approve"));

    expect(resolve).toHaveBeenCalledWith(JSON.stringify({ approved: true }));
  });

  it("resolves rejection with the typed feedback", () => {
    mockUseAgent.mockReturnValue({ agent: fakeAgent() });
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    const resolve = vi.fn();
    const payload = {
      action: "write_report",
      filename: "x.md",
      content: "# Draft",
      attempt: 1,
      max_attempts: 3,
    };
    act(() => {
      interruptHandler?.({ event: { value: JSON.stringify(payload) }, resolve });
    });

    fireEvent.click(screen.getByText("mock-reject"));

    expect(resolve).toHaveBeenCalledWith(JSON.stringify({ approved: false, feedback: "more detail" }));
  });

  it("computes stats from agent state at the moment the interrupt fires", () => {
    mockUseAgent.mockReturnValue({
      agent: fakeAgent({
        state: {
          messages: [
            { name: "supervisor", content: "..." },
            { name: "research_agent", content: "..." },
            { name: "supervisor", content: "..." },
          ],
          research_findings: [{ source: "a", content: "b", relevance_score: 0.5 }],
          analytics_results: [{ metric: "mean", value: 1, detail: "d" }],
        },
      }),
    });
    let fakeNow = 1_000;
    const nowSpy = vi.spyOn(Date, "now").mockImplementation(() => fakeNow);

    render(<ComparePage />);
    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    fakeNow = 6_000;
    const payload = {
      action: "write_report",
      filename: "x.md",
      content: "# Draft",
      attempt: 1,
      max_attempts: 3,
    };
    act(() => {
      interruptHandler?.({ event: { value: JSON.stringify(payload) }, resolve: vi.fn() });
    });

    expect(resultDetailsProps.current?.stats).toEqual({
      analyticsResults: 1,
      supervisorVisits: 2,
      elapsedMs: 5000,
    });
    nowSpy.mockRestore();
  });

  it("stays recoverable when the run errors without ever drafting", () => {
    mockUseAgent.mockReturnValue({
      agent: fakeAgent({
        isRunning: false,
        state: { messages: [], research_findings: [], analytics_results: [], error: "boom" },
      }),
    });
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    expect(screen.getByText("Run")).not.toBeDisabled();
    expect(screen.getByText(/ended without a draft/i)).toBeInTheDocument();
  });

  it("recovers after switching providers following a failed run", () => {
    const openaiAgent = fakeAgent({
      isRunning: false,
      state: { messages: [], research_findings: [], analytics_results: [], error: "boom" },
    });
    const anthropicAgent = fakeAgent();
    mockUseAgent.mockImplementation((args: { agentId: string }) =>
      args.agentId === "anthropicAgent" ? { agent: anthropicAgent } : { agent: openaiAgent },
    );
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    // OpenAI run failed before ever drafting: stuck-state recovery kicks in.
    expect(screen.getByText("Run")).not.toBeDisabled();
    expect(screen.getByText(/ended without a draft/i)).toBeInTheDocument();

    // Switching providers must actually unstick the page, not just leave `runFailed`
    // true for the old provider while `running` stays wedged for the new one.
    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "anthropic" } });

    expect(screen.getByLabelText("Provider")).not.toBeDisabled();
    expect(screen.getByText("Run")).not.toBeDisabled();
    expect(screen.queryByText(/ended without a draft/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("Run"));
    expect(anthropicAgent.runAgent).toHaveBeenCalledTimes(1);
  });

  it("disables the provider select and Run button while a run is genuinely in progress", () => {
    mockUseAgent.mockReturnValue({ agent: fakeAgent() });
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    expect(screen.getByLabelText("Provider")).toBeDisabled();
    expect(screen.getByText("Run")).toBeDisabled();
  });

  it("unsticks the page via Reset when the run never errors and never drafts", () => {
    // Transport-level failure: the browser never reaches the deployment, so `isRunning`
    // is whatever the agent reports and `state.error` is never set -- `runFailed` can't
    // fire. Reset is the always-available way out.
    mockUseAgent.mockReturnValue({ agent: fakeAgent() });
    render(<ComparePage />);

    fireEvent.change(screen.getByLabelText(/objective/i), { target: { value: "Assess pricing" } });
    fireEvent.click(screen.getByText("Run"));

    expect(screen.getByLabelText("Provider")).toBeDisabled();
    expect(screen.getByText("Run")).toBeDisabled();
    expect(screen.queryByText(/ended without a draft/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("Reset"));

    expect(screen.getByText("Run")).not.toBeDisabled();
    expect(screen.getByLabelText("Provider")).not.toBeDisabled();
    expect(screen.queryByText("Reset")).not.toBeInTheDocument();
  });
});
