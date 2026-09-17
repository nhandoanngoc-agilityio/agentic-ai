import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ResultDetails } from "./ResultDetails";
import type { ReportInterruptPayload } from "@/lib/types";

const payload: ReportInterruptPayload = {
  action: "write_report",
  filename: "x.md",
  content: "# Draft report",
  attempt: 1,
  max_attempts: 3,
};

describe("ResultDetails", () => {
  it("renders the draft, findings table, analytics table, and stats", () => {
    render(
      <ResultDetails
        payload={payload}
        findings={[{ source: "doc-a", content: "Acme charges $49/seat", relevance_score: 0.9 }]}
        results={[{ metric: "mean", value: 49, detail: "mean([49])" }]}
        stats={{ analyticsResults: 1, supervisorVisits: 2, elapsedMs: 5000 }}
        onApprove={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText(/Draft report/)).toBeInTheDocument();
    expect(screen.getByText("doc-a")).toBeInTheDocument();
    expect(screen.getByText(/Acme charges \$49\/seat/)).toBeInTheDocument();
    expect(screen.getByText("mean")).toBeInTheDocument();
    expect(screen.getByText(/Research findings: 1/)).toBeInTheDocument();
    expect(screen.getByText(/Analytics results: 1/)).toBeInTheDocument();
    expect(screen.getByText(/Supervisor visits: 2/)).toBeInTheDocument();
  });

  it("calls onApprove when Approve is clicked", () => {
    const onApprove = vi.fn();
    render(
      <ResultDetails
        payload={payload}
        findings={[]}
        results={[]}
        stats={{ analyticsResults: 0, supervisorVisits: 0, elapsedMs: 0 }}
        onApprove={onApprove}
        onReject={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByText("Approve"));

    expect(onApprove).toHaveBeenCalledTimes(1);
  });

  it("calls onReject with the typed feedback when Reject is clicked", () => {
    const onReject = vi.fn();
    render(
      <ResultDetails
        payload={payload}
        findings={[]}
        results={[]}
        stats={{ analyticsResults: 0, supervisorVisits: 0, elapsedMs: 0 }}
        onApprove={vi.fn()}
        onReject={onReject}
      />,
    );

    fireEvent.change(screen.getByLabelText("Rejection feedback"), {
      target: { value: "Add more detail" },
    });
    fireEvent.click(screen.getByText("Reject with feedback"));

    expect(onReject).toHaveBeenCalledWith("Add more detail");
  });

  it("calls onReject with undefined when feedback is left empty", () => {
    const onReject = vi.fn();
    render(
      <ResultDetails
        payload={payload}
        findings={[]}
        results={[]}
        stats={{ analyticsResults: 0, supervisorVisits: 0, elapsedMs: 0 }}
        onApprove={vi.fn()}
        onReject={onReject}
      />,
    );

    fireEvent.click(screen.getByText("Reject with feedback"));

    expect(onReject).toHaveBeenCalledWith(undefined);
  });

  it("shows a message and no table when findings/results are empty", () => {
    render(
      <ResultDetails
        payload={payload}
        findings={[]}
        results={[]}
        stats={{ analyticsResults: 0, supervisorVisits: 0, elapsedMs: 0 }}
        onApprove={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText("No research findings.")).toBeInTheDocument();
    expect(screen.getByText("No analytics results.")).toBeInTheDocument();
  });
});
