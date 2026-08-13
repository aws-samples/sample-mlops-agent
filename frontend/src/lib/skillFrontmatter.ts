// Frontmatter parser for skill SKILL.md files — mirrors the Python
// agent/skills_loader parser. Frontmatter (name/description/tools/model) is
// optional; when absent, name falls back to the provided id and description to
// the first prose line. Used by the Skills editor to display + validate skills.

export interface ParsedSkill {
  name: string;
  description: string;
  tools: string[];
  model: string | null;
  body: string;
}

const FRONTMATTER_RE = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;

function firstProseLine(text: string): string {
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s || s.startsWith("#") || s.startsWith(">")) continue;
    return s;
  }
  return "";
}

export function parseSkill(md: string, fallbackName: string): ParsedSkill {
  const m = FRONTMATTER_RE.exec(md);
  const meta: Record<string, string> = {};
  let body = md;
  if (m) {
    body = m[2];
    for (const line of m[1].split("\n")) {
      if (!line.trim()) continue;
      const idx = line.indexOf(":");
      if (idx === -1) continue;
      const key = line.slice(0, idx).trim();
      const val = line.slice(idx + 1).trim();
      if (["name", "description", "tools", "model"].includes(key))
        meta[key] = val;
    }
  }
  const tools = meta.tools
    ? meta.tools
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
    : [];
  return {
    name: meta.name || fallbackName,
    description: meta.description || firstProseLine(body),
    tools,
    model: meta.model || null,
    body,
  };
}

// Returns an error string if the content is invalid for the expected skill id,
// or null when valid. Frontmatter-less content is valid (name derives from id);
// when frontmatter IS present, its name must match the expected id.
export function validateSkill(md: string, expectedName: string): string | null {
  const m = FRONTMATTER_RE.exec(md);
  if (!m) return null;
  const parsed = parseSkill(md, expectedName);
  const hasNameKey = /^\s*name\s*:/m.test(m[1]);
  if (hasNameKey && parsed.name !== expectedName) {
    return `Frontmatter name "${parsed.name}" must match the skill id "${expectedName}".`;
  }
  return null;
}
