#!/usr/bin/env node
// Stages agent skill docs + architecture diagram into frontend/.agent-skills/
// so the CDK FrontendAsset zip (which only includes frontend/) can carry them
// into CodeBuild. Run by CDK at synth time before packaging the frontend.
import {
  cpSync,
  mkdirSync,
  rmSync,
  statSync,
  copyFileSync,
  existsSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const frontendRoot = resolve(__dirname, "..");
const repoRoot = resolve(frontendRoot, "..");
const srcSkills = join(repoRoot, "agent", ".claude", "skills");
const srcArch = join(
  repoRoot,
  "assets",
  "images",
  "architecture",
  "architecture.png",
);
// Gateway skill tool specs — staged so build-tools-manifest.mjs can read them
// inside CodeBuild (which only sees the zipped frontend/ tree).
const srcSpecs = join(repoRoot, "lambda", "skills");
const specDirs = [
  "sagemaker",
  "huggingface",
  "git",
  "mlflow",
  "slurm",
  "web_search",
  "hyperpod",
];
const stageDir = join(frontendRoot, ".agent-skills");
const destSkills = join(stageDir, "skills");
const destArch = join(stageDir, "architecture.png");
const destSpecs = join(stageDir, "tool-specs");

function fail(msg) {
  console.error(`[copy-agent-skills] ${msg}`);
  process.exit(1);
}

if (!statSync(srcSkills, { throwIfNoEntry: false })?.isDirectory())
  fail(`Missing source skills directory: ${srcSkills}`);
if (!statSync(srcArch, { throwIfNoEntry: false })?.isFile())
  fail(`Missing architecture.png: ${srcArch}`);

if (existsSync(stageDir)) rmSync(stageDir, { recursive: true, force: true });
mkdirSync(stageDir, { recursive: true });
cpSync(srcSkills, destSkills, { recursive: true });
copyFileSync(srcArch, destArch);

// Stage each skill's tool_spec.json under tool-specs/<dir>/tool_spec.json,
// preserving the directory layout build-tools-manifest.mjs expects.
mkdirSync(destSpecs, { recursive: true });
for (const dir of specDirs) {
  const src = join(srcSpecs, dir, "tool_spec.json");
  if (!statSync(src, { throwIfNoEntry: false })?.isFile())
    fail(`Missing tool spec: ${src}`);
  mkdirSync(join(destSpecs, dir), { recursive: true });
  copyFileSync(src, join(destSpecs, dir, "tool_spec.json"));
}
console.log(
  `[copy-agent-skills] staged ${srcSkills} -> ${destSkills} + ${specDirs.length} tool specs`,
);
