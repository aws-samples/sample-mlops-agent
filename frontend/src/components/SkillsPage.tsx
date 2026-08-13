import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import Cards from "@cloudscape-design/components/cards";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import Link from "@cloudscape-design/components/link";
import Spinner from "@cloudscape-design/components/spinner";
import Alert from "@cloudscape-design/components/alert";
import SpaceBetween from "@cloudscape-design/components/space-between";
import { useState } from "react";
import { useSkillsStore } from "../hooks/useSkillsStore";
import { SkillEditor } from "./SkillEditor";
import { SystemPromptEditor } from "./SystemPromptEditor";

interface Props {
  sessionBucket: string;
  skillsPrefix: string;
}

export function SkillsPage({ sessionBucket, skillsPrefix }: Props) {
  const store = useSkillsStore(sessionBucket, skillsPrefix);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Live agent skills stored in S3 — edits apply within ~60s, no redeploy."
          counter={`(${store.skills.length})`}
          actions={
            <Button
              iconName="add-plus"
              onClick={() => {
                const name = (
                  window.prompt("New skill id (lowercase, hyphens):") ?? ""
                ).trim();
                if (name && /^[a-z0-9-]+$/.test(name)) {
                  setNewName(name);
                  setCreating(true);
                }
              }}
            >
              New skill
            </Button>
          }
        >
          Skills
        </Header>
      }
    >
      <SpaceBetween size="l">
      <SystemPromptEditor
        sessionBucket={sessionBucket}
        skillsPrefix={skillsPrefix}
      />
      {store.error && (
        <Alert type="error" header="Failed to load skills">
          {store.error}
        </Alert>
      )}
      {creating && (
        <Box padding={{ bottom: "l" }}>
          <SkillEditor
            name={newName}
            initialContent=""
            isNew
            onSave={async (name, content) => {
              await store.save(name, content);
              setCreating(false);
            }}
            onCancel={() => setCreating(false)}
          />
        </Box>
      )}
      {store.loading ? (
        <Box textAlign="center" padding={{ top: "xxxl" }}>
          <Spinner size="large" />
        </Box>
      ) : (
        <Cards
          cardDefinition={{
            header: (s) => (
              <Link href={`#/skills/${s.name}`} fontSize="heading-m">
                {s.name}
              </Link>
            ),
            sections: [
              {
                id: "description",
                content: (s) => (
                  <Box color="text-body-secondary">{s.description}</Box>
                ),
              },
            ],
          }}
          items={store.skills}
          trackBy="name"
          cardsPerRow={[
            { cards: 1 },
            { minWidth: 640, cards: 2 },
            { minWidth: 1024, cards: 3 },
          ]}
          empty={
            <Box textAlign="center" padding={{ top: "xxxl", bottom: "l" }}>
              No skills found.
            </Box>
          }
        />
      )}
      </SpaceBetween>
    </ContentLayout>
  );
}
