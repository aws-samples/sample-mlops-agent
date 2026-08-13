// Browser-direct S3 CRUD for the Skills editor, using Cognito Identity Pool
// credentials (same pattern as useJobsTable / lib/s3). Skills live at
// s3://<sessionBucket>/<prefix><name>/SKILL.md. system-prompt.md under the
// prefix is NOT a skill and is excluded from listings.
import { useCallback, useEffect, useState } from "react";
import {
  S3Client,
  ListObjectsV2Command,
  GetObjectCommand,
  PutObjectCommand,
  DeleteObjectCommand,
} from "@aws-sdk/client-s3";
import { getAWSCredentials } from "../lib/credentials";
import { parseSkill, type ParsedSkill } from "../lib/skillFrontmatter";

const REGION = import.meta.env.VITE_AWS_REGION ?? "us-east-1";

// Skills that must always exist — deleting them would break the agent.
export const CORE_SKILLS = ["planning"];

async function client(): Promise<S3Client> {
  const credentials = await getAWSCredentials();
  if (!credentials) throw new Error("Not authenticated");
  return new S3Client({ region: REGION, credentials });
}

function skillNameFromKey(key: string, prefix: string): string | null {
  const rel = key.startsWith(prefix) ? key.slice(prefix.length) : key;
  const parts = rel.split("/");
  // Only <name>/SKILL.md is a skill; system-prompt.md (no slash) is excluded.
  if (parts.length === 2 && parts[1] === "SKILL.md") return parts[0];
  return null;
}

export async function listSkills(
  bucket: string,
  prefix: string,
): Promise<ParsedSkill[]> {
  const c = await client();
  const res = await c.send(
    new ListObjectsV2Command({ Bucket: bucket, Prefix: prefix }),
  );
  const names: string[] = [];
  for (const obj of res.Contents ?? []) {
    const n = obj.Key ? skillNameFromKey(obj.Key, prefix) : null;
    if (n) names.push(n);
  }
  const out: ParsedSkill[] = [];
  for (const name of names) {
    const got = await c.send(
      new GetObjectCommand({
        Bucket: bucket,
        Key: `${prefix}${name}/SKILL.md`,
      }),
    );
    const text = await (
      got.Body as { transformToString: () => Promise<string> }
    ).transformToString();
    out.push(parseSkill(text, name));
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

export async function getSkill(
  bucket: string,
  prefix: string,
  name: string,
): Promise<string> {
  const c = await client();
  const got = await c.send(
    new GetObjectCommand({ Bucket: bucket, Key: `${prefix}${name}/SKILL.md` }),
  );
  return (
    got.Body as { transformToString: () => Promise<string> }
  ).transformToString();
}

export async function saveSkill(
  bucket: string,
  prefix: string,
  name: string,
  content: string,
): Promise<void> {
  const c = await client();
  await c.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: `${prefix}${name}/SKILL.md`,
      Body: content,
      ContentType: "text/markdown",
    }),
  );
}

// Read the current system prompt from <prefix>system-prompt.md. Returns "" when
// the object does not exist yet (mirrors the agent's load_system_prompt, which
// returns None and falls back to the baked-in claude_code preset). Used by the
// Skills-tab editor and to seed the Evals optimizer's "current prompt" input.
export async function loadSystemPrompt(
  bucket: string,
  prefix: string,
): Promise<string> {
  const c = await client();
  try {
    const got = await c.send(
      new GetObjectCommand({
        Bucket: bucket,
        Key: `${prefix}system-prompt.md`,
      }),
    );
    return (
      got.Body as { transformToString: () => Promise<string> }
    ).transformToString();
  } catch (e) {
    // Absent object is the expected "not set yet" state — return empty. Any
    // other S3 error (access denied, network) must surface, not be masked.
    if (e instanceof Error && e.name === "NoSuchKey") return "";
    throw e;
  }
}

// The agent reads the system prompt from <prefix>system-prompt.md (a FLAT
// object, not a <name>/SKILL.md dir). Used by the Evals "apply optimization".
export async function saveSystemPrompt(
  bucket: string,
  prefix: string,
  content: string,
): Promise<void> {
  const c = await client();
  await c.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: `${prefix}system-prompt.md`,
      Body: content,
      ContentType: "text/markdown",
    }),
  );
}

export async function deleteSkill(
  bucket: string,
  prefix: string,
  name: string,
  currentNames: string[],
): Promise<void> {
  if (CORE_SKILLS.includes(name)) {
    throw new Error(`"${name}" is a core skill and cannot be deleted.`);
  }
  if (currentNames.length <= 1) {
    throw new Error(
      "Cannot delete the last remaining skill — at least one must exist.",
    );
  }
  const c = await client();
  await c.send(
    new DeleteObjectCommand({
      Bucket: bucket,
      Key: `${prefix}${name}/SKILL.md`,
    }),
  );
}

export interface SkillsStore {
  skills: ParsedSkill[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  save: (name: string, content: string) => Promise<void>;
  remove: (name: string) => Promise<void>;
}

// React hook wrapper over the standalone functions for the Skills tab.
export function useSkillsStore(bucket: string, prefix: string): SkillsStore {
  const [skills, setSkills] = useState<ParsedSkill[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!bucket) return;
    setLoading(true);
    setError(null);
    try {
      setSkills(await listSkills(bucket, prefix));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [bucket, prefix]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const save = useCallback(
    async (name: string, content: string) => {
      await saveSkill(bucket, prefix, name, content);
      await refresh();
    },
    [bucket, prefix, refresh],
  );

  const remove = useCallback(
    async (name: string) => {
      await deleteSkill(
        bucket,
        prefix,
        name,
        skills.map((s) => s.name),
      );
      await refresh();
    },
    [bucket, prefix, refresh, skills],
  );

  return { skills, loading, error, refresh, save, remove };
}
