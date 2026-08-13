import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Spinner from "@cloudscape-design/components/spinner";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Textarea from "@cloudscape-design/components/textarea";
import { useCallback, useEffect, useState } from "react";
import { loadSystemPrompt, saveSystemPrompt } from "../hooks/useSkillsStore";

interface Props {
  sessionBucket: string;
  skillsPrefix: string;
}

// Views/edits the agent's dynamic system prompt (<prefix>system-prompt.md in
// S3). This is NOT a skill (no frontmatter), so it uses a plain textarea rather
// than SkillEditor. When empty, the agent falls back to the baked-in
// claude_code preset; setting it here is also what makes the Evals "System
// Prompt" optimization work (the optimizer needs a non-empty prompt to refine).
//
// Read-only first: the current prompt is shown as text with an "Edit" button;
// the textarea + Save/Cancel only appear after the user clicks Edit.
export function SystemPromptEditor({ sessionBucket, skillsPrefix }: Props) {
  // `saved` holds the last-persisted content (the read-only view + Cancel
  // baseline); `draft` is the in-progress edit.
  const [savedContent, setSavedContent] = useState("");
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justSaved, setJustSaved] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSavedContent(await loadSystemPrompt(sessionBucket, skillsPrefix));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [sessionBucket, skillsPrefix]);

  useEffect(() => {
    void load();
  }, [load]);

  function startEdit() {
    setDraft(savedContent);
    setJustSaved(false);
    setError(null);
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
    setError(null);
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      await saveSystemPrompt(sessionBucket, skillsPrefix, draft);
      setSavedContent(draft);
      setEditing(false);
      setJustSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Container
      header={
        <Header
          variant="h2"
          description="Agent system prompt (S3) — applied within ~60s, no redeploy. Empty = built-in default."
          actions={
            !loading &&
            !editing && (
              <Button iconName="edit" onClick={startEdit}>
                Edit
              </Button>
            )
          }
        >
          System prompt
        </Header>
      }
    >
      {loading ? (
        <Box textAlign="center" padding={{ top: "l", bottom: "l" }}>
          <Spinner />
        </Box>
      ) : editing ? (
        <SpaceBetween size="m">
          {error && <Alert type="error">{error}</Alert>}
          <Textarea
            value={draft}
            onChange={({ detail }) => setDraft(detail.value)}
            rows={16}
            spellcheck={false}
            placeholder="No custom system prompt set — the agent uses its built-in default. Enter markdown here to override it."
            ariaLabel="Edit agent system prompt"
          />
          <SpaceBetween size="xs" direction="horizontal">
            <Button variant="primary" loading={saving} onClick={save}>
              Save
            </Button>
            <Button onClick={cancelEdit} disabled={saving}>
              Cancel
            </Button>
          </SpaceBetween>
        </SpaceBetween>
      ) : (
        <SpaceBetween size="m">
          {error && <Alert type="error">{error}</Alert>}
          {justSaved && <StatusIndicator type="success">Saved</StatusIndicator>}
          {savedContent.trim() ? (
            <Box variant="code" fontSize="body-s">
              <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                {savedContent}
              </pre>
            </Box>
          ) : (
            <Box color="text-body-secondary">
              No custom system prompt set — the agent uses its built-in default.
              Click Edit to add one.
            </Box>
          )}
        </SpaceBetween>
      )}
    </Container>
  );
}
