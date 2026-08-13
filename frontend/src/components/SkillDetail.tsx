import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Spinner from "@cloudscape-design/components/spinner";
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  getSkill,
  saveSkill,
  deleteSkill,
  listSkills,
  CORE_SKILLS,
} from "../hooks/useSkillsStore";
import { parseSkill } from "../lib/skillFrontmatter";
import { SkillEditor } from "./SkillEditor";

interface Props {
  skillId: string;
  sessionBucket: string;
  skillsPrefix: string;
}

export function SkillDetail({ skillId, sessionBucket, skillsPrefix }: Props) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getSkill(sessionBucket, skillsPrefix, skillId)
      .then((c) => !cancelled && setContent(c))
      .catch(
        (e) =>
          !cancelled && setError(e instanceof Error ? e.message : String(e)),
      )
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [skillId, sessionBucket, skillsPrefix]);

  if (loading) {
    return (
      <ContentLayout header={<Header variant="h1">{skillId}</Header>}>
        <Box textAlign="center" padding={{ top: "xxxl" }}>
          <Spinner size="large" />
        </Box>
      </ContentLayout>
    );
  }

  if (error || content === null) {
    return (
      <ContentLayout header={<Header variant="h1">Skill not found</Header>}>
        <Alert
          type="error"
          action={<Button href="#/skills">Back to Skills</Button>}
        >
          Could not load skill <strong>{skillId}</strong>. {error}
        </Alert>
      </ContentLayout>
    );
  }

  const parsed = parseSkill(content, skillId);

  if (editing) {
    return (
      <ContentLayout
        header={<Header variant="h1">Editing {parsed.name}</Header>}
      >
        <SkillEditor
          name={skillId}
          initialContent={content}
          onSave={async (name, c) => {
            await saveSkill(sessionBucket, skillsPrefix, name, c);
            setContent(c);
            setEditing(false);
          }}
          onCancel={() => setEditing(false)}
        />
      </ContentLayout>
    );
  }

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description={parsed.description}
          actions={
            <SpaceBetween size="xs" direction="horizontal">
              <Button iconName="edit" onClick={() => setEditing(true)}>
                Edit
              </Button>
              <Button
                iconName="remove"
                disabled={CORE_SKILLS.includes(skillId)}
                onClick={async () => {
                  if (!window.confirm(`Delete skill "${skillId}"?`)) return;
                  try {
                    // Fetch current names so the "last remaining skill" guard
                    // is enforced with real data, not a sentinel.
                    const names = (
                      await listSkills(sessionBucket, skillsPrefix)
                    ).map((s) => s.name);
                    await deleteSkill(
                      sessionBucket,
                      skillsPrefix,
                      skillId,
                      names,
                    );
                    window.location.hash = "#/skills";
                  } catch (e) {
                    setError(e instanceof Error ? e.message : String(e));
                  }
                }}
              >
                Delete
              </Button>
              <Button href="#/skills" iconName="arrow-left">
                Back to Skills
              </Button>
            </SpaceBetween>
          }
        >
          {parsed.name}
        </Header>
      }
    >
      <Box padding="l">
        <div className="skill-markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {parsed.body}
          </ReactMarkdown>
        </div>
      </Box>
    </ContentLayout>
  );
}
