// useSkillsStore: browser-direct S3 CRUD for the skills editor, using Cognito
// Identity Pool creds (same pattern as useJobsTable). We mock the S3 client and
// the creds helper. Delete is guarded: refuses to drop below one skill and
// refuses to delete a core skill (planning).
import { beforeEach, describe, expect, it, vi } from "vitest";

const send = vi.fn();
// All of these are constructed with `new`, so each must be a real constructor
// that stamps a discriminator + copies the input for assertions.
vi.mock("@aws-sdk/client-s3", () => ({
  S3Client: class {
    send = send;
  },
  ListObjectsV2Command: class {
    constructor(i: object) {
      Object.assign(this, { __t: "list" }, i);
    }
  },
  GetObjectCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "get" }, i);
    }
  },
  PutObjectCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "put" }, i);
    }
  },
  DeleteObjectCommand: class {
    constructor(i: object) {
      Object.assign(this, { __t: "del" }, i);
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
  listSkills,
  saveSkill,
  deleteSkill,
  loadSystemPrompt,
  saveSystemPrompt,
} from "../useSkillsStore";

const BUCKET = "sess-bucket";
const PREFIX = "skills/";

function bodyStream(text: string) {
  return { transformToString: async () => text };
}

beforeEach(() => {
  send.mockReset();
});

describe("listSkills", () => {
  it("lists skills/<name>/SKILL.md and parses each", async () => {
    send.mockImplementation((cmd: { __t: string; Key?: string }) => {
      if (cmd.__t === "list")
        return Promise.resolve({
          Contents: [
            { Key: "skills/git/SKILL.md" },
            { Key: "skills/system-prompt.md" },
          ],
        });
      return Promise.resolve({
        Body: bodyStream("# Git Skill\n\nGit via MCP.\n"),
      });
    });
    const skills = await listSkills(BUCKET, PREFIX);
    expect(skills.map((s) => s.name)).toEqual(["git"]); // system-prompt.md excluded
    expect(skills[0].description).toBe("Git via MCP.");
  });
});

describe("saveSkill", () => {
  it("PutObject at skills/<name>/SKILL.md", async () => {
    send.mockResolvedValue({});
    await saveSkill(BUCKET, PREFIX, "git", "# Git\n");
    const putCall = send.mock.calls.find((c) => c[0].__t === "put")!;
    expect(putCall[0].Key).toBe("skills/git/SKILL.md");
    expect(putCall[0].Bucket).toBe(BUCKET);
  });
});

describe("deleteSkill guard", () => {
  it("refuses to delete a core skill", async () => {
    await expect(
      deleteSkill(BUCKET, PREFIX, "planning", ["planning", "git"]),
    ).rejects.toThrow(/core/i);
  });

  it("refuses to delete the last remaining skill", async () => {
    await expect(deleteSkill(BUCKET, PREFIX, "git", ["git"])).rejects.toThrow(
      /last|at least one/i,
    );
  });

  it("deletes an ordinary skill", async () => {
    send.mockResolvedValue({});
    await deleteSkill(BUCKET, PREFIX, "git", ["git", "sagemaker"]);
    const delCall = send.mock.calls.find((c) => c[0].__t === "del")!;
    expect(delCall[0].Key).toBe("skills/git/SKILL.md");
  });
});

describe("system prompt", () => {
  it("loads <prefix>system-prompt.md", async () => {
    send.mockResolvedValue({ Body: bodyStream("You are the agent.") });
    const text = await loadSystemPrompt(BUCKET, PREFIX);
    const getCall = send.mock.calls.find((c) => c[0].__t === "get")!;
    expect(getCall[0].Key).toBe("skills/system-prompt.md");
    expect(text).toBe("You are the agent.");
  });

  it("returns '' when the object does not exist (NoSuchKey)", async () => {
    const err = new Error("not found");
    err.name = "NoSuchKey";
    send.mockRejectedValue(err);
    expect(await loadSystemPrompt(BUCKET, PREFIX)).toBe("");
  });

  it("propagates non-NoSuchKey errors", async () => {
    const err = new Error("access denied");
    err.name = "AccessDenied";
    send.mockRejectedValue(err);
    await expect(loadSystemPrompt(BUCKET, PREFIX)).rejects.toThrow(
      /access denied/i,
    );
  });

  it("saves <prefix>system-prompt.md", async () => {
    send.mockResolvedValue({});
    await saveSystemPrompt(BUCKET, PREFIX, "# New prompt");
    const putCall = send.mock.calls.find((c) => c[0].__t === "put")!;
    expect(putCall[0].Key).toBe("skills/system-prompt.md");
    expect(putCall[0].Body).toBe("# New prompt");
  });
});
