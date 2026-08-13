import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SkillEditor } from "../SkillEditor";

const VALID = "# Git\n\nGit via MCP.\n";
const MISMATCH = "---\nname: other\n---\n# X\n";

function textarea(): HTMLTextAreaElement {
  return screen.getByRole("textbox") as HTMLTextAreaElement;
}

describe("SkillEditor", () => {
  it("renders the skill content in a textarea", () => {
    render(
      <SkillEditor
        name="git"
        initialContent={VALID}
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    expect(textarea().value).toContain("Git via MCP.");
  });

  it("calls onSave with (name, content) for valid content", () => {
    const onSave = vi.fn();
    render(
      <SkillEditor
        name="git"
        initialContent={VALID}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Save"));
    expect(onSave).toHaveBeenCalledWith("git", VALID);
  });

  it("blocks Save and shows an error when frontmatter name mismatches", () => {
    const onSave = vi.fn();
    render(
      <SkillEditor
        name="git"
        initialContent={VALID}
        onSave={onSave}
        onCancel={vi.fn()}
      />,
    );
    fireEvent.change(textarea(), { target: { value: MISMATCH } });
    fireEvent.click(screen.getByText("Save"));
    expect(onSave).not.toHaveBeenCalled();
    expect(screen.getByText(/must match the skill id/i)).toBeInTheDocument();
  });

  it("seeds a template for a new skill", () => {
    render(
      <SkillEditor
        name="newskill"
        initialContent=""
        isNew
        onSave={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    expect(textarea().value).toContain("name: newskill");
  });
});
