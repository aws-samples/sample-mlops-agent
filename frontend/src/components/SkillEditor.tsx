import Alert from "@cloudscape-design/components/alert";
import Button from "@cloudscape-design/components/button";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Textarea from "@cloudscape-design/components/textarea";
import { useState } from "react";
import { validateSkill } from "../lib/skillFrontmatter";

interface Props {
  name: string;
  initialContent: string;
  isNew?: boolean;
  onSave: (name: string, content: string) => void;
  onCancel: () => void;
}

function template(name: string): string {
  return `---\nname: ${name}\ndescription: One-line description of this skill.\n---\n# ${name} Skill\n\nDescribe what this skill does and the MCP tools it uses.\n`;
}

// Presentational editor: validates frontmatter on every change (Save is
// disabled while invalid) and hands (name, content) to onSave. New skills are
// seeded from a template so the frontmatter name matches the id.
export function SkillEditor({
  name,
  initialContent,
  isNew,
  onSave,
  onCancel,
}: Props) {
  const [content, setContent] = useState(
    isNew && !initialContent ? template(name) : initialContent,
  );
  const error = validateSkill(content, name);

  return (
    <SpaceBetween size="m">
      {error && <Alert type="error">{error}</Alert>}
      <Textarea
        value={content}
        onChange={({ detail }) => setContent(detail.value)}
        rows={24}
        spellcheck={false}
        ariaLabel={`Edit skill ${name}`}
      />
      <SpaceBetween size="xs" direction="horizontal">
        <Button
          variant="primary"
          disabled={error !== null}
          onClick={() => {
            if (validateSkill(content, name) === null) onSave(name, content);
          }}
        >
          Save
        </Button>
        <Button onClick={onCancel}>Cancel</Button>
      </SpaceBetween>
    </SpaceBetween>
  );
}
