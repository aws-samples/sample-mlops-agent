import ExpandableSection from "@cloudscape-design/components/expandable-section";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import type { ToolEntry } from "../types";

function prettyJson(src: string): string {
  try {
    return JSON.stringify(JSON.parse(src), null, 2);
  } catch {
    return src;
  }
}

interface Props {
  entry: ToolEntry;
}

export function ToolCallCard({ entry }: Props) {
  const statusType =
    entry.result === undefined
      ? "in-progress"
      : entry.isError
        ? "error"
        : "success";

  const header = (
    <span
      style={{
        fontFamily: "monospace",
        fontSize: 12,
        color: "var(--color-text-body-secondary, #687078)",
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <StatusIndicator type={statusType}>{entry.name}</StatusIndicator>
      <span>{entry.status}</span>
    </span>
  );

  const showBody = entry.result !== undefined || entry.args;

  return (
    <div data-testid="tool-call-card" style={{ margin: "4px 0" }}>
      <ExpandableSection variant="inline" headerText={header}>
        {showBody && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 4 }}>
                Arguments
              </div>
              <pre
                style={{
                  margin: 0,
                  padding: 8,
                  fontSize: 12,
                  background: "var(--color-background-layout-main, #f2f3f3)",
                  borderRadius: 4,
                  overflow: "auto",
                  maxHeight: 200,
                }}
              >
                {prettyJson(entry.args)}
              </pre>
            </div>
            {entry.result !== undefined && (
              <div>
                <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 4 }}>
                  Output
                </div>
                <pre
                  style={{
                    margin: 0,
                    padding: 8,
                    fontSize: 12,
                    background: "var(--color-background-layout-main, #f2f3f3)",
                    color: entry.isError
                      ? "var(--color-text-status-error, #d13212)"
                      : undefined,
                    borderRadius: 4,
                    overflow: "auto",
                    maxHeight: 320,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {entry.result || "(no output)"}
                </pre>
                {entry.truncated && (
                  <div style={{ fontSize: 11, opacity: 0.6, marginTop: 4 }}>
                    Output truncated at 8 KB
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </ExpandableSection>
    </div>
  );
}
