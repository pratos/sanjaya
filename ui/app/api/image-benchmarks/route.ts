import { readdir, readFile } from "fs/promises";
import { join } from "path";
import { NextResponse } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

interface TraceEvent {
  kind: string;
  timestamp: number;
  [key: string]: unknown;
}

const KIND_MAP: Record<string, string> = {
  "sanjaya.completion_start": "run_start",
  "sanjaya.completion_end": "run_end",
  "sanjaya.iteration_start": "iteration_start",
  "sanjaya.iteration_end": "iteration_end",
  "sanjaya.root_llm_call_start": "root_response_start",
  "sanjaya.root_llm_call_end": "root_response",
  "sanjaya.code_execution_start": "code_instruction",
  "sanjaya.code_execution_end": "code_execution",
  "sanjaya.tool_call_start": "tool_call_start",
  "sanjaya.tool_call_end": "tool_call",
  "sanjaya.sub_llm_call.regular_start": "sub_llm_start",
  "sanjaya.sub_llm_call.regular_end": "sub_llm",
  "sanjaya.sub_llm_call.image_vision_start": "vision_start",
  "sanjaya.sub_llm_call.image_vision_end": "vision",
  "sanjaya.sub_llm_call.image_compare_start": "vision_start",
  "sanjaya.sub_llm_call.image_compare_end": "vision",
  "sanjaya.schema_generation_start": "schema_generation_start",
  "sanjaya.schema_generation_end": "schema_generation",
  "sanjaya.critic_evaluation": "critic_evaluation",
};

function normalizeTraceEvents(events: TraceEvent[]) {
  return events.map((e) => {
    const { kind: rawKind, timestamp, ...payload } = e;
    return { kind: KIND_MAP[rawKind] ?? rawKind, timestamp, payload };
  });
}

interface RawImageResult {
  name: string;
  image_paths: string[];
  question: string;
  answer_text: string;
  answer_data: Record<string, unknown> | null;
  ground_truth?: string;
  iterations: number;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  wall_time_s: number;
  evidence_count: number;
  evidence_sources: string[];
  error?: string;
  trace_events?: TraceEvent[];
}

function toCamelCase(raw: RawImageResult, promptId: number) {
  return {
    promptId,
    promptName: raw.name ?? `prompt_${promptId}`,
    imagePaths: raw.image_paths ?? [],
    question: raw.question,
    answerText: raw.answer_text ?? "",
    answerData: raw.answer_data ?? null,
    groundTruth: raw.ground_truth ?? null,
    iterations: raw.iterations ?? 0,
    costUsd: raw.cost_usd ?? 0,
    inputTokens: raw.input_tokens ?? 0,
    outputTokens: raw.output_tokens ?? 0,
    wallTimeS: raw.wall_time_s ?? 0,
    evidenceCount: raw.evidence_count ?? 0,
    evidenceSources: raw.evidence_sources ?? [],
    error: raw.error,
    traceEvents: raw.trace_events ? normalizeTraceEvents(raw.trace_events) : null,
  };
}

async function discoverImageVersionDirs(): Promise<Record<string, string>> {
  let entries: string[];
  try {
    entries = await readdir(DATA_DIR);
  } catch {
    return {};
  }

  const dirs = entries.filter((e) => e.startsWith("demo_image_results")).sort();
  const versions: Record<string, string> = {};
  for (const dir of dirs) {
    const match = dir.match(/^demo_image_results(?:_v(\d+))?$/);
    if (match) {
      const label = match[1] ? `v${match[1]}` : "v1";
      versions[label] = dir;
    }
  }
  return versions;
}

async function loadImageVersion(version: string, dirName: string) {
  const dir = join(DATA_DIR, dirName);
  const results: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];

  let files: string[];
  try {
    const entries = await readdir(dir);
    files = entries.filter((f) => f.endsWith(".json") && f !== "summary.json");
  } catch {
    return results;
  }

  let promptId = 1;
  for (const file of files.sort()) {
    try {
      const raw = JSON.parse(await readFile(join(dir, file), "utf-8")) as RawImageResult;
      results.push({ version, data: toCamelCase(raw, promptId++) });
    } catch { /* skip */ }
  }

  return results;
}

export async function GET() {
  const versionDirs = await discoverImageVersionDirs();
  const allResults: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];

  for (const [version, dirName] of Object.entries(versionDirs)) {
    const results = await loadImageVersion(version, dirName);
    allResults.push(...results);
  }

  // Group by promptName (since IDs are per-version)
  const byName = new Map<string, Record<string, ReturnType<typeof toCamelCase>>>();
  for (const { version, data } of allResults) {
    const key = data.promptName;
    if (!byName.has(key)) byName.set(key, {});
    byName.get(key)![version] = data;
  }

  let promptId = 1;
  const prompts = Array.from(byName.entries())
    .map(([name, versions]) => {
      const sortedVersions = Object.keys(versions).sort((a, b) => {
        const numA = parseInt(a.replace("v", ""), 10) || 0;
        const numB = parseInt(b.replace("v", ""), 10) || 0;
        return numB - numA;
      });
      let bestVersion = sortedVersions[0] ?? "v1";
      for (const v of sortedVersions) {
        if (versions[v] && !versions[v].error) { bestVersion = v; break; }
      }
      const best = versions[bestVersion] ?? Object.values(versions)[0];
      return {
        promptId: promptId++,
        promptName: name,
        question: best.question,
        versions,
        bestVersion,
      };
    });

  // Summary
  let totalCostUsd = 0, totalWallTimeS = 0;
  const versionSet = new Set<string>();
  for (const p of prompts) {
    for (const [v, data] of Object.entries(p.versions)) {
      versionSet.add(v);
      if (v === p.bestVersion && !data.error) {
        totalCostUsd += data.costUsd;
        totalWallTimeS += data.wallTimeS;
      }
    }
  }
  const sortedVersions = Array.from(versionSet).sort((a, b) => {
    return (parseInt(a.replace("v", ""), 10) || 0) - (parseInt(b.replace("v", ""), 10) || 0);
  });

  return NextResponse.json({
    prompts,
    summary: {
      totalPrompts: prompts.length,
      versions: sortedVersions,
      totalCostUsd: Math.round(totalCostUsd * 10000) / 10000,
      totalWallTimeS: Math.round(totalWallTimeS * 10) / 10,
      latestVersion: sortedVersions[sortedVersions.length - 1] ?? "v1",
    },
  });
}
