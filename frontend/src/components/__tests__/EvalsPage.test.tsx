import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SummaryPanel } from "../evals/SummaryPanel";
import { BatchPanel } from "../evals/BatchPanel";
import { parseHash } from "../../lib/router";

describe("router", () => {
  it("parses #/evals", () => {
    expect(parseHash("#/evals")).toEqual({ name: "evals" });
  });
});

describe("SummaryPanel", () => {
  it("renders a stat tile per evaluator", () => {
    render(
      <SummaryPanel
        loading={false}
        error={null}
        data={{
          "Builtin.GoalSuccessRate": { avg: 0.8, count: 5 },
          "Builtin.Helpfulness": { avg: 0.9, count: 5 },
        }}
      />,
    );
    expect(screen.getByText("Builtin.GoalSuccessRate")).toBeInTheDocument();
    expect(screen.getByText(/0\.8/)).toBeInTheDocument();
  });

  it("shows an empty-state when no sessions scored", () => {
    render(
      <SummaryPanel
        loading={false}
        error={null}
        data={{ "Builtin.GoalSuccessRate": { avg: null, count: 0 } }}
      />,
    );
    expect(screen.getByText(/no sessions scored yet/i)).toBeInTheDocument();
  });

  it("shows a calm session-expired notice, not the raw AWS error", () => {
    render(
      <SummaryPanel
        loading={false}
        error={
          'CognitoIdentity.GetId failed: {"__type":"NotAuthorizedException","message":"Invalid login token. Token expired"}'
        }
        data={{}}
      />,
    );
    expect(screen.getByText(/signing you back in/i)).toBeInTheDocument();
    // The raw exception text must not be shown to the user.
    expect(screen.queryByText(/NotAuthorizedException/)).toBeNull();
  });

  it("still shows genuine (non-auth) errors verbatim", () => {
    render(
      <SummaryPanel
        loading={false}
        error="CloudWatch GetMetricData throttled"
        data={{}}
      />,
    );
    expect(
      screen.getByText(/cloudwatch getmetricdata throttled/i),
    ).toBeInTheDocument();
  });
});

describe("BatchPanel", () => {
  it("runs a batch eval on button click", () => {
    const run = vi.fn();
    render(
      <BatchPanel
        status={null}
        results={null}
        running={false}
        error={null}
        run={run}
      />,
    );
    fireEvent.click(screen.getByText(/run batch evaluation/i));
    expect(run).toHaveBeenCalled();
  });

  it("shows polling state while running", () => {
    render(
      <BatchPanel
        status="IN_PROGRESS"
        results={null}
        running={true}
        error={null}
        run={vi.fn()}
      />,
    );
    expect(screen.getByText(/in_progress|running/i)).toBeInTheDocument();
  });
});
