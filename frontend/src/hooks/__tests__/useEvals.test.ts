// Evals data hooks: CloudWatch GetMetricData for the 7-day summary, and the
// bedrock-agentcore data-plane client for batch eval + recommendation. All AWS
// clients are mocked (real constructors, since the code uses `new`).
import { beforeEach, describe, expect, it, vi } from "vitest";

const cwSend = vi.fn();
const bacSend = vi.fn();

vi.mock("@aws-sdk/client-cloudwatch", () => ({
  CloudWatchClient: class {
    send = cwSend;
  },
  GetMetricDataCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "getMetricData" }, i);
    }
  },
}));
vi.mock("@aws-sdk/client-bedrock-agentcore", () => ({
  BedrockAgentCoreClient: class {
    send = bacSend;
  },
  StartBatchEvaluationCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "startBatch" }, i);
    }
  },
  GetBatchEvaluationCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "getBatch" }, i);
    }
  },
  StartRecommendationCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "startRec" }, i);
    }
  },
  GetRecommendationCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "getRec" }, i);
    }
  },
}));
vi.mock("../../lib/credentials", () => ({
  getAWSCredentials: vi.fn(async () => ({
    accessKeyId: "AK",
    secretAccessKey: "SK",
    sessionToken: "T",
  })),
}));

import {
  fetchSummary,
  startBatchEval,
  getBatchEval,
  startRecommendation,
  getRecommendation,
} from "../useEvals";

const CFG = {
  agentRuntimeName: "sample_mlops_agent",
  onlineEvalConfigName: "sample_mlops_agent_online_eval",
  evalResultsLogGroupPrefix: "/aws/bedrock-agentcore/evaluations/results/x",
  runtimeLogGroup:
    "/aws/bedrock-agentcore/runtimes/sample_mlops_agent-xez02QGVzr-DEFAULT",
  runtimeLogGroupArn:
    "arn:aws:logs:us-east-1:381492284087:log-group:/aws/bedrock-agentcore/runtimes/sample_mlops_agent-xez02QGVzr-DEFAULT:*",
};
const EVALUATORS = [
  "Builtin.GoalSuccessRate",
  "Builtin.Helpfulness",
  "Builtin.Correctness",
];

beforeEach(() => {
  cwSend.mockReset();
  bacSend.mockReset();
});

describe("fetchSummary", () => {
  it("queries GetMetricData with the <runtime>.DEFAULT service filter", async () => {
    cwSend.mockResolvedValue({
      MetricDataResults: [
        { Id: "avg_0", Timestamps: [new Date()], Values: [0.8] },
        { Id: "cnt_0", Timestamps: [new Date()], Values: [5] },
      ],
    });
    const res = await fetchSummary(EVALUATORS, 7);
    const call = cwSend.mock.calls[0][0];
    expect(call.__t).toBe("getMetricData");
    const q = JSON.stringify(call.MetricDataQueries);
    expect(q).toContain("sample_mlops_agent.DEFAULT");
    expect(q).toContain("Bedrock-AgentCore/Evaluations");
    expect(res["Builtin.GoalSuccessRate"].avg).toBe(0.8);
  });
});

describe("batch eval", () => {
  it("start passes the runtime service + evaluators and returns the ARN", async () => {
    bacSend.mockResolvedValue({
      batchEvaluationId: "b1",
      batchEvaluationArn: "arn:aws:bedrock-agentcore:us-east-1:123:batch/b1",
    });
    const out = await startBatchEval(CFG, EVALUATORS);
    // The optimizer needs the ARN, not the id/name.
    expect(out.batchEvaluationArn).toBe(
      "arn:aws:bedrock-agentcore:us-east-1:123:batch/b1",
    );
    const call = bacSend.mock.calls[0][0];
    expect(call.__t).toBe("startBatch");
    expect(call.dataSourceConfig.cloudWatchLogs.serviceNames).toEqual([
      "sample_mlops_agent.DEFAULT",
    ]);
    // Must read the REAL app log group (spans + LLO), not /runtimes/<name>.
    expect(call.dataSourceConfig.cloudWatchLogs.logGroupNames).toContain(
      "/aws/bedrock-agentcore/runtimes/sample_mlops_agent-xez02QGVzr-DEFAULT",
    );
    // Batch name must be hyphen-free (API pattern).
    expect(call.batchEvaluationName).toMatch(/^[a-zA-Z][a-zA-Z0-9_]*$/);
    expect(
      call.evaluators.map((e: { evaluatorId: string }) => e.evaluatorId),
    ).toEqual(EVALUATORS);
  });

  it("get returns status + results + ARN", async () => {
    bacSend.mockResolvedValue({
      status: "COMPLETED",
      evaluationResults: [],
      batchEvaluationArn: "arn:aws:bedrock-agentcore:us-east-1:123:batch/b1",
    });
    const out = await getBatchEval("b1");
    expect(bacSend.mock.calls[0][0].__t).toBe("getBatch");
    expect(out.status).toBe("COMPLETED");
    expect(out.batchEvaluationArn).toBe(
      "arn:aws:bedrock-agentcore:us-east-1:123:batch/b1",
    );
  });
});

describe("recommendation", () => {
  it("system-prompt start with a batch arn calls StartRecommendation", async () => {
    bacSend.mockResolvedValue({ recommendationId: "r1" });
    await startRecommendation(CFG, "SYSTEM_PROMPT_RECOMMENDATION", {
      currentPrompt: "You are the agent.",
      batchEvaluationArn: "arn:aws:...:batch/b1",
    });
    const call = bacSend.mock.calls[0][0];
    expect(call.__t).toBe("startRec");
    expect(call.type).toBe("SYSTEM_PROMPT_RECOMMENDATION");
    expect(
      call.recommendationConfig.systemPromptRecommendationConfig.agentTraces
        .batchEvaluation.batchEvaluationArn,
    ).toBe("arn:aws:...:batch/b1");
  });

  it("tool-description start sends a non-empty logGroupArns + tools", async () => {
    bacSend.mockResolvedValue({ recommendationId: "r2" });
    const tools = [
      {
        toolName: "git-skill___commit_experiment",
        toolDescription: { text: "Commit." },
      },
    ];
    await startRecommendation(CFG, "TOOL_DESCRIPTION_RECOMMENDATION", {
      tools,
    });
    const call = bacSend.mock.calls[0][0];
    expect(call.type).toBe("TOOL_DESCRIPTION_RECOMMENDATION");
    const cfg = call.recommendationConfig.toolDescriptionRecommendationConfig;
    const arns = cfg.agentTraces.cloudwatchLogs.logGroupArns;
    // API rejects an empty list — must carry the runtime log group ARN.
    expect(arns.length).toBeGreaterThanOrEqual(1);
    expect(arns[0]).toBe(CFG.runtimeLogGroupArn);
    // API also rejects an empty tools list — must carry >=1 tool.
    const sent = cfg.toolDescription.toolDescriptionText.tools;
    expect(sent.length).toBeGreaterThanOrEqual(1);
    expect(sent[0].toolName).toBe("git-skill___commit_experiment");
  });

  it("system-prompt start throws when the current prompt is empty", async () => {
    bacSend.mockResolvedValue({ recommendationId: "r4" });
    await expect(
      startRecommendation(CFG, "SYSTEM_PROMPT_RECOMMENDATION", {
        currentPrompt: "   ",
        batchEvaluationArn: "arn:aws:...:batch/b1",
      }),
    ).rejects.toThrow(/no system prompt is set/i);
    expect(bacSend).not.toHaveBeenCalled();
  });

  it("tool-description start throws when tools list is empty", async () => {
    bacSend.mockResolvedValue({ recommendationId: "r3" });
    await expect(
      startRecommendation(CFG, "TOOL_DESCRIPTION_RECOMMENDATION", {}),
    ).rejects.toThrow(/non-empty tools list/);
    expect(bacSend).not.toHaveBeenCalled();
  });

  it("get surfaces the nested failure errorMessage", async () => {
    bacSend.mockResolvedValue({
      status: "FAILED",
      recommendationResult: {
        systemPromptRecommendationResult: {
          errorMessage: "No sessions were identified from input agent traces.",
        },
      },
    });
    const out = await getRecommendation("r5");
    expect(out.status).toBe("FAILED");
    expect(out.errorMessage).toBe(
      "No sessions were identified from input agent traces.",
    );
  });
});
