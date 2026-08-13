import { useCallback, useRef, useState } from "react";
import { SignatureV4 } from "@smithy/signature-v4";
import { Sha256 } from "@aws-crypto/sha256-browser";
import { getAWSCredentials, getCognitoSub } from "../lib/credentials";
import type { TimelineEntry, MessageEntry, ToolEntry } from "../types";

interface UseChatOptions {
  agentCoreEndpoint: string;
  threadId: string;
  onRunFinished?: () => void;
}

function buildInvocationParts(endpoint: string): {
  hostname: string;
  path: string;
} {
  // Build the path with ARN percent-encoded (colons → %3A, slashes inside ARN → %2F),
  // matching what boto3 produces. We keep path as a plain string (never go through
  // URL.pathname) to avoid browser normalisation decoding %3A back to :.
  const m = endpoint.match(/^https:\/\/([^/]+)(\/runtimes\/)(.+)$/);
  if (m) {
    return {
      hostname: m[1],
      path: `${m[2]}${encodeURIComponent(m[3])}/invocations`,
    };
  }
  const fallback = new URL(endpoint);
  return {
    hostname: fallback.hostname,
    path: `${fallback.pathname}/invocations`,
  };
}

async function signedFetch(
  endpoint: string,
  body: object,
  region: string,
  signal?: AbortSignal,
): Promise<Response> {
  const credentials = await getAWSCredentials();
  if (!credentials)
    throw new Error("No AWS credentials — is the user signed in?");
  const { hostname, path } = buildInvocationParts(endpoint);
  const bodyStr = JSON.stringify(body);
  const signer = new SignatureV4({
    credentials,
    region,
    service: "bedrock-agentcore",
    sha256: Sha256,
  });
  const signed = await signer.sign({
    method: "POST",
    hostname,
    path,
    protocol: "https:",
    headers: { host: hostname, "content-type": "application/json" },
    body: bodyStr,
  });
  const res = await fetch(`https://${hostname}${path}`, {
    method: "POST",
    headers: signed.headers as Record<string, string>,
    body: bodyStr,
    signal,
  });
  if (!res.ok) {
    const errBody = await res.text().catch(() => "(unreadable)");
    throw new Error(`AgentCore returned ${res.status}: ${errBody}`);
  }
  return res;
}

export function useChat({
  agentCoreEndpoint,
  threadId,
  onRunFinished,
}: UseChatOptions) {
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timelineRef = useRef<TimelineEntry[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const region = import.meta.env.VITE_AWS_REGION ?? "us-east-1";

  const updateTimeline = useCallback(
    (updater: (prev: TimelineEntry[]) => TimelineEntry[]) => {
      setTimeline((prev) => {
        const next = updater(prev);
        timelineRef.current = next;
        return next;
      });
    },
    [],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleEvent = useCallback(
    (
      event: { type?: string; [k: string]: unknown },
      assistantId: string,
      appendAssistant: (delta: string) => string,
    ): void => {
      if (
        event.type === "TEXT_MESSAGE_CHUNK" &&
        typeof event.delta === "string"
      ) {
        const next = appendAssistant(event.delta);
        updateTimeline((prev) => {
          const exists = prev.find(
            (e) => e.kind === "message" && e.id === assistantId,
          );
          if (exists) {
            return prev.map((e) =>
              e.kind === "message" && e.id === assistantId
                ? { ...e, content: next }
                : e,
            );
          }
          const msg: MessageEntry = {
            kind: "message",
            id: assistantId,
            role: "assistant",
            content: next,
          };
          return [...prev, msg];
        });
        return;
      }
      if (
        event.type === "TOOL_CALL_CHUNK" &&
        typeof event.tool_call_name === "string"
      ) {
        const entry: ToolEntry = {
          kind: "tool",
          id: (event.tool_call_id as string) ?? crypto.randomUUID(),
          step: (event.step as number) ?? 0,
          name: event.tool_call_name,
          status: (event.status as string) ?? event.tool_call_name,
          args: (event.delta as string) ?? "{}",
        };
        updateTimeline((prev) => [...prev, entry]);
        return;
      }
      if (
        event.type === "TOOL_CALL_RESULT" &&
        typeof event.tool_call_id === "string"
      ) {
        updateTimeline((prev) => {
          const idx = prev.findIndex(
            (e) => e.kind === "tool" && e.id === event.tool_call_id,
          );
          if (idx === -1) {
            const placeholder: ToolEntry = {
              kind: "tool",
              id: event.tool_call_id as string,
              step: (event.step as number) ?? 0,
              name: "(unknown)",
              status: "(unknown)",
              args: "{}",
              result: (event.content as string) ?? "",
              truncated: !!event.truncated,
              isError: !!event.is_error,
            };
            return [...prev, placeholder];
          }
          return prev.map((e, i) =>
            i === idx
              ? {
                  ...e,
                  result: (event.content as string) ?? "",
                  truncated: !!event.truncated,
                  isError: !!event.is_error,
                }
              : e,
          );
        });
      }
    },
    [updateTimeline],
  );

  const send = useCallback(
    async (text: string, runFinishedCb?: () => void) => {
      const userMsg: MessageEntry = {
        kind: "message",
        role: "user",
        id: crypto.randomUUID(),
        content: text,
      };
      updateTimeline((prev) => [...prev, userMsg]);
      setStreaming(true);
      setError(null);

      const controller = new AbortController();
      abortRef.current = controller;

      const runId = crypto.randomUUID();
      const assistantId = crypto.randomUUID();
      let assistantContent = "";

      const historyForRequest = timelineRef.current
        .filter((e): e is MessageEntry => e.kind === "message")
        .map((m) => ({ id: m.id, role: m.role, content: m.content }));

      const body = {
        session_id: threadId,
        run_id: runId,
        user_id: getCognitoSub(),
        messages: [
          ...historyForRequest,
          { id: userMsg.id, role: "user", content: text },
        ],
      };

      try {
        const res = await signedFetch(
          agentCoreEndpoint,
          body,
          region,
          controller.signal,
        );
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop()!;
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const jsonStr = line.startsWith("data: ") ? line.slice(6) : line;
              if (!jsonStr.trim()) continue;
              const event = JSON.parse(jsonStr);
              handleEvent(event, assistantId, (delta) => {
                assistantContent += delta;
                return assistantContent;
              });
              if (event.type === "RUN_FINISHED")
                (runFinishedCb ?? onRunFinished)?.();
            } catch {
              /* skip malformed lines */
            }
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          // user interrupted
        } else {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setStreaming(false);
      }
    },
    [
      agentCoreEndpoint,
      threadId,
      region,
      updateTimeline,
      onRunFinished,
      handleEvent,
    ],
  );

  const reset = useCallback(() => {
    setTimeline([]);
    timelineRef.current = [];
    setStreaming(false);
    setError(null);
  }, []);

  return { timeline, streaming, error, send, stop, reset };
}
