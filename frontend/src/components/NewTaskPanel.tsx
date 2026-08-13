import { useEffect, useRef, useState } from "react";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import Header from "@cloudscape-design/components/header";
import Spinner from "@cloudscape-design/components/spinner";
import { useChat } from "../hooks/useChat";
import { MarkdownMessage } from "./MarkdownMessage";
import { ToolCallCard } from "./ToolCallCard";
import { STARTER_TILES, StarterTile } from "./StarterTile";

interface Props {
  agentCoreEndpoint: string;
  threadId: string;
  onSubmit: (text: string) => void;
  onRunFinished: (threadId: string) => void;
  onClose: () => void;
}

export function NewTaskPanel({
  agentCoreEndpoint,
  threadId,
  onSubmit,
  onRunFinished,
  onClose,
}: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const { timeline, streaming, error, send, stop } = useChat({
    agentCoreEndpoint,
    threadId,
    onRunFinished: () => onRunFinished(threadId),
  });

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [timeline.length]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    onSubmit(text);
    await send(text);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        boxSizing: "border-box",
      }}
    >
      {/* Page header */}
      <div style={{ padding: "20px 40px 0 40px" }}>
        <Header
          variant="h1"
          description="Describe what you want to do. The agent will start immediately when you send a message."
          actions={
            <Button
              iconName="close"
              variant="icon"
              onClick={onClose}
              ariaLabel="Close"
            />
          }
        >
          New Task
        </Header>
      </div>

      {/* Centered chat column */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          padding: "24px 40px 24px 40px",
        }}
      >
        <div
          style={{
            flex: 1,
            maxWidth: "800px",
            width: "100%",
            margin: "0 auto",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
            overflow: "hidden",
          }}
        >
          {timeline.length === 0 && (
            <div
              style={{ display: "flex", flexDirection: "column", gap: "8px" }}
            >
              <Box color="text-body-secondary">Start with a suggestion:</Box>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, 1fr)",
                  gap: "12px",
                }}
              >
                {STARTER_TILES.map((tile) => (
                  <StarterTile
                    key={tile.title}
                    {...tile}
                    onSelect={(prompt) => setInput(prompt)}
                  />
                ))}
              </div>
            </div>
          )}

          {timeline.length > 0 && (
            <div
              ref={scrollRef}
              style={{
                flex: 1,
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: "12px",
              }}
            >
              {timeline.map((e) => {
                if (e.kind === "tool")
                  return <ToolCallCard key={e.id} entry={e} />;
                return (
                  <div
                    key={e.id}
                    style={{
                      display: "flex",
                      justifyContent:
                        e.role === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    {e.role === "user" ? (
                      <div
                        style={{
                          background:
                            "var(--color-background-notification-blue, #0972d3)",
                          color: "#ffffff",
                          borderRadius: "8px",
                          padding: "10px 14px",
                          maxWidth: "75%",
                          fontSize: "14px",
                          lineHeight: "22px",
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-word",
                        }}
                      >
                        {e.content}
                      </div>
                    ) : (
                      <div style={{ maxWidth: "75%" }}>
                        <MarkdownMessage content={e.content} />
                      </div>
                    )}
                  </div>
                );
              })}
              {streaming && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "4px 0",
                  }}
                >
                  <Spinner size="normal" />
                  <Box color="text-body-secondary" fontSize="body-s">
                    <span>Thinking…</span>
                  </Box>
                </div>
              )}
            </div>
          )}

          {error && <Box color="text-status-error">{error}</Box>}

          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend();
                }
              }}
              placeholder="Describe your task…"
              disabled={streaming}
              rows={3}
              style={{
                width: "100%",
                padding: "10px 14px",
                border:
                  "1px solid var(--color-border-control-default, #adb5bd)",
                borderRadius: "8px",
                fontSize: "14px",
                lineHeight: "22px",
                fontFamily:
                  'var(--font-family-base, "Amazon Ember", "Helvetica Neue", Roboto, Arial, sans-serif)',
                resize: "none",
                boxSizing: "border-box",
                color: "var(--color-text-body-default)",
                background: "var(--color-background-input-default)",
              }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              {streaming ? (
                <Button variant="normal" onClick={stop}>
                  Stop
                </Button>
              ) : (
                <Button
                  variant="primary"
                  onClick={() => void handleSend()}
                  disabled={!input.trim()}
                >
                  Send
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
