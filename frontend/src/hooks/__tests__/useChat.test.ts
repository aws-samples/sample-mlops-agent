import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useChat } from "../useChat";

vi.mock("../../lib/credentials", () => ({
  getAWSCredentials: async () => ({
    accessKeyId: "AK",
    secretAccessKey: "SK",
    sessionToken: "tok",
  }),
  getCognitoSub: () => "sub-1",
}));

function sseResponse(events: object[]): Response {
  const body =
    events.map((e) => `data: ${JSON.stringify(e)}\n`).join("") + "\n";
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      sseResponse([
        { type: "TEXT_MESSAGE_CHUNK", delta: "hi " },
        {
          type: "TOOL_CALL_CHUNK",
          tool_call_id: "tu1",
          tool_call_name: "Bash",
          delta: JSON.stringify({ command: "ls" }),
          status: "Running command: ls",
          step: 1,
        },
        {
          type: "TOOL_CALL_RESULT",
          tool_call_id: "tu1",
          content: "a\nb\n",
          truncated: false,
          is_error: false,
          step: 1,
        },
        { type: "TEXT_MESSAGE_CHUNK", delta: "there" },
        { type: "RUN_FINISHED" },
      ]),
    ),
  );
});

describe("useChat timeline", () => {
  it("interleaves messages and tool entries, patches result in place", async () => {
    const { result } = renderHook(() =>
      useChat({
        agentCoreEndpoint: "https://x/runtimes/arn",
        threadId: "t1",
      }),
    );
    await act(async () => {
      await result.current.send("go");
    });
    await waitFor(() => expect(result.current.streaming).toBe(false));
    const tl = result.current.timeline;
    expect(tl.map((e) => e.kind)).toEqual(["message", "message", "tool"]);
    const tool = tl[2] as {
      kind: "tool";
      id: string;
      result?: string;
      truncated?: boolean;
      isError?: boolean;
    };
    expect(tool.id).toBe("tu1");
    expect(tool.result).toBe("a\nb\n");
    expect(tool.truncated).toBe(false);
    expect(tool.isError).toBe(false);
    const assistant = tl[1] as { kind: "message"; content: string };
    expect(assistant.content).toBe("hi there");
  });
});
