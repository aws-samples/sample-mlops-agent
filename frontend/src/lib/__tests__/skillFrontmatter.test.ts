// Mirrors the Python agent/skills_loader parser: frontmatter is optional; when
// absent, name falls back to the provided id and description to the first prose
// line. validateSkill enforces the frontmatter name matches the expected id.
import { describe, expect, it } from "vitest";
import { parseSkill, validateSkill } from "../skillFrontmatter";

const WITH_FM = `---
name: sagemaker
description: SageMaker training + deployment.
tools: a-tool, b-tool
model: claude-opus-4-5
---
# SageMaker Skill

Body.
`;

const NO_FM = `# Git Skill

All git operations via MCP.
`;

describe("parseSkill", () => {
  it("parses frontmatter fields", () => {
    const s = parseSkill(WITH_FM, "sagemaker");
    expect(s.name).toBe("sagemaker");
    expect(s.description).toBe("SageMaker training + deployment.");
    expect(s.tools).toEqual(["a-tool", "b-tool"]);
    expect(s.model).toBe("claude-opus-4-5");
    expect(s.body.trim().startsWith("# SageMaker Skill")).toBe(true);
  });

  it("falls back when no frontmatter", () => {
    const s = parseSkill(NO_FM, "git");
    expect(s.name).toBe("git");
    expect(s.description).toBe("All git operations via MCP.");
    expect(s.tools).toEqual([]);
    expect(s.model).toBeNull();
  });
});

describe("validateSkill", () => {
  it("passes when frontmatter name matches", () => {
    expect(validateSkill(WITH_FM, "sagemaker")).toBeNull();
  });

  it("returns an error when frontmatter name mismatches", () => {
    const err = validateSkill(WITH_FM, "git");
    expect(err).toMatch(/name/i);
  });

  it("passes frontmatter-less content (name derives from id)", () => {
    expect(validateSkill(NO_FM, "git")).toBeNull();
  });
});
