import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ToolCallCard } from "../ToolCallCard";
import type { ToolEntry } from "../../types";

const base: ToolEntry = {
  kind: "tool",
  id: "tu1",
  step: 1,
  name: "Bash",
  status: "Running command: ls",
  args: '{"command":"ls"}',
};

// Cloudscape's inline ExpandableSection hides content when collapsed.
// Click the header button to expand before asserting body content.
function expand(container: HTMLElement) {
  const button = container.querySelector<HTMLButtonElement>(
    '[role="button"], button',
  );
  if (!button) throw new Error("expand button not found");
  fireEvent.click(button);
}

describe("ToolCallCard", () => {
  it("renders in-progress state header", () => {
    render(<ToolCallCard entry={base} />);
    expect(screen.getByText(/Running command: ls/)).toBeInTheDocument();
    expect(screen.getByText(/Bash/)).toBeInTheDocument();
  });

  it("renders success + truncated when expanded", () => {
    const { container } = render(
      <ToolCallCard entry={{ ...base, result: "a\nb\n", truncated: true }} />,
    );
    expand(container);
    expect(screen.getByText(/Output truncated at 8 KB/)).toBeInTheDocument();
  });

  it("renders error state output when expanded", () => {
    const { container } = render(
      <ToolCallCard entry={{ ...base, result: "boom", isError: true }} />,
    );
    expand(container);
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });
});
