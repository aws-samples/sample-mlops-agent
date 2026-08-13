import { useEffect, useState } from "react";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand } from "@aws-sdk/lib-dynamodb";
import { getAWSCredentials } from "../lib/credentials";
import type { TimelineEntry } from "../types";

const REGION = import.meta.env.VITE_AWS_REGION ?? "us-east-1";
const TABLE_NAME =
  import.meta.env.VITE_JOBS_TABLE_NAME ?? "sample-mlops-agent-metadata";

function isTimelineEntry(e: unknown): e is TimelineEntry {
  if (!e || typeof e !== "object") return false;
  const v = e as Record<string, unknown>;
  if (v.kind === "message") {
    return (
      typeof v.id === "string" &&
      (v.role === "user" || v.role === "assistant") &&
      typeof v.content === "string"
    );
  }
  if (v.kind === "tool") {
    return (
      typeof v.id === "string" &&
      typeof v.name === "string" &&
      typeof v.status === "string" &&
      typeof v.args === "string"
    );
  }
  return false;
}

async function fetchTimeline(threadId: string): Promise<TimelineEntry[]> {
  const credentials = await getAWSCredentials();
  if (!credentials) return [];
  const client = DynamoDBDocumentClient.from(
    new DynamoDBClient({ region: REGION, credentials }),
  );
  const result = await client.send(
    new GetCommand({ TableName: TABLE_NAME, Key: { task_id: threadId } }),
  );
  const raw = result.Item?.timeline;
  if (!raw) return [];
  try {
    const parsed: unknown = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isTimelineEntry) as TimelineEntry[];
  } catch {
    return [];
  }
}

// Poll interval for stored chat history. The agent persists the full
// timeline via _upsert_chat_row at the END of a run — so if the user
// navigates from NewTaskPanel to TaskDetailPanel mid-run, the first fetch
// can miss in-flight turns. Polling catches up within a few seconds once
// the run closes and the timeline blob gets rewritten.
const POLL_MS = 5_000;

export function useChatHistory(threadId: string, enabled = true) {
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!enabled || !threadId) return;
    let cancelled = false;
    const refresh = async (isFirst: boolean) => {
      if (isFirst) setLoading(true);
      try {
        const next = await fetchTimeline(threadId);
        if (!cancelled) setTimeline(next);
      } catch {
        if (!cancelled && isFirst) setTimeline([]);
      } finally {
        if (isFirst && !cancelled) setLoading(false);
      }
    };
    void refresh(true);
    const id = setInterval(() => {
      void refresh(false);
    }, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [threadId, enabled]);

  return { timeline, loading };
}
