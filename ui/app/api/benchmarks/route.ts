import { readdir, readFile } from "fs/promises";
import { join } from "path";
import { NextResponse } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

/** Auto-discover demo_results* directories and assign version labels. */
async function discoverVersionDirs(): Promise<Record<string, string>> {
  let entries: string[];
  try {
    entries = await readdir(DATA_DIR);
  } catch {
    return {};
  }

  const dirs = entries
    .filter((e) => e.startsWith("demo_results"))
    .sort(); // demo_results, demo_results_v2, demo_results_v3, ...

  const versions: Record<string, string> = {};
  for (const dir of dirs) {
    // demo_results -> v1, demo_results_v2 -> v2, demo_results_v3 -> v3, etc.
    const match = dir.match(/^demo_results(?:_v(\d+))?$/);
    if (match) {
      const label = match[1] ? `v${match[1]}` : "v1";
      versions[label] = dir;
    }
  }

  return versions;
}

const VIDEO_PATHS: Record<string, string> = {
  podcast: "youtube/5RAFKES5J6E.mp4",
  curry: "youtube/zVg3FvMuDlw.mp4",
  mkbhd: "youtube/rng_yUSwrgU.mp4",
  football: "youtube/9gyv2xh7qQw.mp4",
  tech_talk: "youtube/qdfwmYTO0Aw.mp4",
};

interface TraceEvent {
  kind: string;
  timestamp: number;
  [key: string]: unknown;
}

interface RawResult {
  prompt_id: number;
  prompt_name: string;
  video_key: string;
  question: string;
  answer_text: string;
  answer_data: Record<string, unknown> | null;
  iterations: number;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  wall_time_s: number;
  evidence_count: number;
  evidence_sources: string[];
  subtitle: {
    had_existing_subtitle: boolean;
    subtitle_generated: boolean;
    subtitle_source: string;
  };
  error?: string;
  trace_events?: TraceEvent[];
}

/** Map internal tracer span names to frontend event names. */
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
  "sanjaya.sub_llm_call.vision_start": "vision_start",
  "sanjaya.sub_llm_call.vision_end": "vision",
  "sanjaya.sub_llm_call.caption_frames_start": "vision_start",
  "sanjaya.sub_llm_call.caption_frames_end": "vision",
  "sanjaya.schema_generation_start": "schema_generation_start",
  "sanjaya.schema_generation_end": "schema_generation",
  "sanjaya.critic_evaluation": "critic_evaluation",
};

function normalizeTraceEvents(events: TraceEvent[]): Array<{ kind: string; timestamp: number; payload: Record<string, unknown> }> {
  return events.map((e) => {
    const { kind: rawKind, timestamp, ...payload } = e;
    return {
      kind: KIND_MAP[rawKind] ?? rawKind,
      timestamp,
      payload,
    };
  });
}

function toCamelCase(raw: RawResult) {
  return {
    promptId: raw.prompt_id,
    promptName: raw.prompt_name,
    videoKey: raw.video_key,
    question: raw.question,
    answerText: raw.answer_text ?? "",
    answerData: raw.answer_data ?? null,
    iterations: raw.iterations ?? 0,
    costUsd: raw.cost_usd ?? 0,
    inputTokens: raw.input_tokens ?? 0,
    outputTokens: raw.output_tokens ?? 0,
    wallTimeS: raw.wall_time_s ?? 0,
    evidenceCount: raw.evidence_count ?? 0,
    evidenceSources: raw.evidence_sources ?? [],
    subtitle: raw.subtitle
      ? {
          hadExistingSubtitle: raw.subtitle.had_existing_subtitle,
          subtitleGenerated: raw.subtitle.subtitle_generated,
          subtitleSource: raw.subtitle.subtitle_source,
        }
      : { hadExistingSubtitle: false, subtitleGenerated: false, subtitleSource: "none" },
    error: raw.error,
    traceEvents: raw.trace_events ? normalizeTraceEvents(raw.trace_events) : null,
  };
}

async function loadVersion(version: string, dirName: string) {
  const dir = join(DATA_DIR, dirName);
  const results: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];

  let files: string[];
  try {
    const entries = await readdir(dir);
    files = entries.filter((f) => f.startsWith("prompt_") && f.endsWith(".json"));
  } catch {
    return results;
  }

  for (const file of files) {
    try {
      const raw = JSON.parse(await readFile(join(dir, file), "utf-8")) as RawResult;
      if (raw.error && !raw.answer_text) {
        // Error-only entry — still include it but mark it
        results.push({ version, data: { ...toCamelCase(raw), error: raw.error } });
      } else {
        results.push({ version, data: toCamelCase(raw) });
      }
    } catch {
      // skip broken files
    }
  }

  return results;
}

/** Trace JSON written by agent._persist_trace for each live run. */
interface TraceJson {
  run_id: string;
  question: string;
  model: string;
  sub_model?: string;
  vision_model?: string;
  answer: string;
  answer_data: Record<string, unknown> | null;
  iterations: number;
  wall_time_s: number;
  cost: {
    total_cost_usd: number;
    total_input_tokens: number;
    total_output_tokens: number;
    elapsed_s?: number;
    calls?: number;
  };
  evidence_count: number;
  events?: TraceEvent[];
  messages?: Array<{ role: string; content: string }>;
}

/** Extract relative video path (e.g. "youtube/abc.mp4") from system message. */
function extractVideoPath(trace: TraceJson): string | null {
  const msgs = trace.messages;
  if (!msgs?.length) return null;
  const content = msgs[0]?.content ?? "";
  const m = content.match(/- video:\s*(.+\.mp4)/);
  if (!m) return null;
  const full = m[1].trim();
  const idx = full.indexOf("/data/");
  if (idx >= 0) return full.slice(idx + 6);
  // Fall back to filename
  const parts = full.split("/");
  return parts[parts.length - 1];
}

interface LiveRunEntry {
  runId: string;
  timestamp: string; // directory name e.g. "20260408-165850"
  model: string;
  videoPath: string | null;
  data: ReturnType<typeof toCamelCase>;
}

/** Scan sanjaya_artifacts/ for live UI runs and convert to the same format as benchmark prompts. */
async function loadLiveRuns(): Promise<LiveRunEntry[]> {
  const artifactsDir = join(process.cwd(), "..", "sanjaya_artifacts");
  const results: LiveRunEntry[] = [];

  let runDirs: string[];
  try {
    runDirs = await readdir(artifactsDir);
  } catch {
    return results;
  }

  // Sort by directory name (timestamp-based) so prompt IDs are chronological
  runDirs.sort();

  for (let i = 0; i < runDirs.length; i++) {
    const runDir = runDirs[i];
    const tracePath = join(artifactsDir, runDir, "trace.json");
    const manifestPath = join(artifactsDir, runDir, "manifest.json");
    try {
      const trace = JSON.parse(await readFile(tracePath, "utf-8")) as TraceJson;
      const promptId = 1000 + i;

      const promptName = trace.question.length > 50
        ? trace.question.slice(0, 50).replace(/\s+\S*$/, "") + "…"
        : trace.question;

      // Read trace events from manifest (same format as benchmark prompt JSONs)
      let traceEvents: TraceEvent[] | undefined;
      try {
        const manifest = JSON.parse(await readFile(manifestPath, "utf-8"));
        const rawEvents: TraceEvent[] = manifest.trace_events ?? [];
        traceEvents = normalizeTraceEvents(rawEvents);
      } catch {
        // Fall back to events from trace.json
        if (trace.events) {
          traceEvents = normalizeTraceEvents(trace.events);
        }
      }

      // Parse answer_data: trace.answer may be a stringified JSON dict
      let answerText = "";
      let answerData = trace.answer_data;
      if (typeof trace.answer === "string") {
        answerText = trace.answer;
        if (!answerData) {
          try { answerData = JSON.parse(trace.answer); } catch { /* not JSON */ }
        }
      } else {
        answerText = JSON.stringify(trace.answer);
      }

      const raw: RawResult = {
        prompt_id: promptId,
        prompt_name: promptName,
        video_key: "live",
        question: trace.question,
        answer_text: answerText,
        answer_data: answerData,
        iterations: trace.iterations,
        cost_usd: trace.cost?.total_cost_usd ?? 0,
        input_tokens: trace.cost?.total_input_tokens ?? 0,
        output_tokens: trace.cost?.total_output_tokens ?? 0,
        wall_time_s: trace.wall_time_s ?? 0,
        evidence_count: trace.evidence_count ?? 0,
        evidence_sources: [],
        subtitle: {
          had_existing_subtitle: false,
          subtitle_generated: false,
          subtitle_source: "unknown",
        },
        trace_events: traceEvents,
      };

      results.push({
        runId: trace.run_id ?? runDir,
        timestamp: runDir,
        model: trace.model ?? "unknown",
        videoPath: extractVideoPath(trace),
        data: toCamelCase(raw),
      });
    } catch {
      // skip runs without trace.json
    }
  }

  return results;
}

export async function GET() {
  // Auto-discover version directories
  const versionDirs = await discoverVersionDirs();

  // Load benchmark versions
  const allResults: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];
  for (const [version, dirName] of Object.entries(versionDirs)) {
    const results = await loadVersion(version, dirName);
    allResults.push(...results);
  }

  // Group by prompt_id
  const byPrompt = new Map<number, Record<string, ReturnType<typeof toCamelCase>>>();
  for (const { version, data } of allResults) {
    if (!byPrompt.has(data.promptId)) {
      byPrompt.set(data.promptId, {});
    }
    byPrompt.get(data.promptId)![version] = data;
  }

  // Build merged prompts
  const prompts = Array.from(byPrompt.entries())
    .sort(([a], [b]) => a - b)
    .map(([promptId, versions]) => {
      // Pick best version: prefer latest (highest number) non-error version
      const sortedVersions = Object.keys(versions).sort((a, b) => {
        const numA = parseInt(a.replace("v", ""), 10) || 0;
        const numB = parseInt(b.replace("v", ""), 10) || 0;
        return numB - numA; // descending
      });
      let bestVersion = sortedVersions[0] ?? "v1";
      for (const v of sortedVersions) {
        if (versions[v] && !versions[v].error) {
          bestVersion = v;
          break;
        }
      }

      const best = versions[bestVersion] ?? Object.values(versions)[0];
      return {
        promptId,
        promptName: best.promptName,
        videoKey: best.videoKey,
        videoPath: VIDEO_PATHS[best.videoKey] ?? null,
        question: best.question,
        versions,
        bestVersion,
      };
    });

  // Summary — compute per-version totals
  let bestCostUsd = 0;
  let bestWallTimeS = 0;
  let v1CostUsd = 0;
  let v1WallTimeS = 0;
  const versionSet = new Set<string>();
  for (const p of prompts) {
    for (const [v, data] of Object.entries(p.versions)) {
      versionSet.add(v);
      if (v === p.bestVersion) {
        bestCostUsd += data.costUsd;
        bestWallTimeS += data.wallTimeS;
      }
      if (v === "v1" && !data.error) {
        v1CostUsd += data.costUsd;
        v1WallTimeS += data.wallTimeS;
      }
    }
  }

  const sortedVersions = Array.from(versionSet).sort((a, b) => {
    const numA = parseInt(a.replace("v", ""), 10) || 0;
    const numB = parseInt(b.replace("v", ""), 10) || 0;
    return numA - numB;
  });
  const latestVersion = sortedVersions[sortedVersions.length - 1] ?? "v1";

  // Load live runs separately
  const liveRunEntries = await loadLiveRuns();

  const liveRunItems = liveRunEntries.map((entry) => ({
    runId: entry.runId,
    timestamp: entry.timestamp,
    model: entry.model,
    prompt: {
      promptId: entry.data.promptId,
      promptName: entry.data.promptName,
      videoKey: entry.data.videoKey,
      videoPath: entry.videoPath,
      question: entry.data.question,
      versions: { live: entry.data },
      bestVersion: "live",
    },
  }));

  // Live runs summary
  let liveCostUsd = 0;
  let liveWallTimeS = 0;
  for (const item of liveRunItems) {
    const d = item.prompt.versions.live;
    if (d && !d.error) {
      liveCostUsd += d.costUsd;
      liveWallTimeS += d.wallTimeS;
    }
  }

  return NextResponse.json({
    prompts,
    summary: {
      totalPrompts: prompts.length,
      versions: sortedVersions,
      totalCostUsd: Math.round(bestCostUsd * 10000) / 10000,
      totalWallTimeS: Math.round(bestWallTimeS * 10) / 10,
      v1CostUsd: Math.round(v1CostUsd * 10000) / 10000,
      v1WallTimeS: Math.round(v1WallTimeS * 10) / 10,
      latestVersion,
    },
    liveRuns: {
      runs: liveRunItems,
      totalRuns: liveRunItems.length,
      totalCostUsd: Math.round(liveCostUsd * 10000) / 10000,
      totalWallTimeS: Math.round(liveWallTimeS * 10) / 10,
    },
    videos: Object.entries(VIDEO_PATHS).map(([key, path]) => ({ key, path })),
  });
}
