import { readdir, readFile } from "fs/promises";
import { join } from "path";
import { NextResponse } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

/** Ground truth baselines for prompts that have known answers. */
const GROUND_TRUTH: Record<number, string> = {
  2: "144 three-pointers",
};

async function directoryHasPromptFiles(relativeDir: string): Promise<boolean> {
  try {
    const files = await readdir(join(DATA_DIR, relativeDir));
    return files.some((f) => f.startsWith("prompt_") && f.endsWith(".json"));
  } catch {
    return false;
  }
}

async function countLvbPromptsInDir(relativeDir: string): Promise<number> {
  try {
    const dir = join(DATA_DIR, relativeDir);
    const files = await readdir(dir);
    const promptFiles = files.filter((f) => f.startsWith("prompt_") && f.endsWith(".json"));

    let count = 0;
    for (const file of promptFiles) {
      try {
        const raw = JSON.parse(await readFile(join(dir, file), "utf-8")) as {
          video_key?: string;
          prompt_id?: number;
        };
        const isLvb =
          (typeof raw.video_key === "string" && raw.video_key.startsWith("lvb_")) ||
          (typeof raw.prompt_id === "number" && raw.prompt_id >= 13);
        if (isLvb) count += 1;
      } catch {
        // skip bad files
      }
    }

    return count;
  } catch {
    return 0;
  }
}

async function bestLvbPromptSubdir(root: string): Promise<string | null> {
  try {
    const entries = await readdir(join(DATA_DIR, root), { withFileTypes: true });
    const subdirs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort().reverse();

    let bestDir: string | null = null;
    let bestCount = 0;

    for (const dir of subdirs) {
      const rel = `${root}/${dir}`;
      const count = await countLvbPromptsInDir(rel);
      if (count > bestCount) {
        bestCount = count;
        bestDir = rel;
      }
    }

    if (bestDir && bestCount > 0) return bestDir;
    return null;
  } catch {
    return null;
  }
}

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

  // Standard flat layout: data/demo_results[_vN]/prompt_*.json
  for (const dir of dirs) {
    const match = dir.match(/^demo_results(?:_v(\d+))?$/);
    if (!match) continue;

    const label = match[1] ? `v${match[1]}` : "v1";
    if (await directoryHasPromptFiles(dir)) {
      versions[label] = dir;
    }
  }

  return versions;
}

async function discoverLvbVersionDirs(): Promise<Record<string, string>> {
  let entries: string[];
  try {
    entries = await readdir(DATA_DIR);
  } catch {
    return {};
  }

  const versions: Record<string, string> = {};

  // Per request: Trinity = v1, Codex53 = v2 for LVB prompts.
  const trinityBest = entries.includes("demo_results_trinity")
    ? await bestLvbPromptSubdir("demo_results_trinity")
    : null;
  if (trinityBest) versions.v1 = trinityBest;

  const codexBest = entries.includes("demo_results_codex53")
    ? await bestLvbPromptSubdir("demo_results_codex53")
    : null;
  if (codexBest) versions.v2 = codexBest;

  return versions;
}

const VIDEO_META: Record<string, { path: string; title: string; channel: string; youtubeId: string; duration: string }> = {
  podcast: {
    path: "youtube/5RAFKES5J6E.mp4",
    title: "RLM Theory Overview feat. Alex L. Zhang",
    channel: "Deep Learning with Yacine",
    youtubeId: "5RAFKES5J6E",
    duration: "2:13:20",
  },
  curry: {
    path: "youtube/zVg3FvMuDlw.mp4",
    title: "Stephen Curry's CRAZIEST Made Threes This Season!",
    channel: "NBA",
    youtubeId: "zVg3FvMuDlw",
    duration: "36:52",
  },
  mkbhd: {
    path: "youtube/rng_yUSwrgU.mp4",
    title: "iPhone 17 Review: No Asterisks!",
    channel: "Marques Brownlee",
    youtubeId: "rng_yUSwrgU",
    duration: "11:20",
  },
  football: {
    path: "youtube/9gyv2xh7qQw.mp4",
    title: "Manchester City v Arsenal | Fourth Round | Emirates FA Cup 2022-23",
    channel: "The Emirates FA Cup",
    youtubeId: "9gyv2xh7qQw",
    duration: "1:42:21",
  },
  tech_talk: {
    path: "youtube/qdfwmYTO0Aw.mp4",
    title: "Prompting Is Becoming a Product Surface",
    channel: "Boundary",
    youtubeId: "qdfwmYTO0Aw",
    duration: "50:14",
  },
  lvb_cooking: {
    path: "longvideobench/videos/1R5uPaL0V-0.mp4",
    title: "LongVideoBench — Cooking",
    channel: "LongVideoBench",
    youtubeId: "1R5uPaL0V-0",
    duration: "16:48",
  },
  lvb_movie: {
    path: "longvideobench/videos/N7RTTiHsSjI.mp4",
    title: "LongVideoBench — Movie Recap",
    channel: "LongVideoBench",
    youtubeId: "N7RTTiHsSjI",
    duration: "08:00",
  },
  lvb_travel: {
    path: "longvideobench/videos/kOZnpwI2hIM.mp4",
    title: "LongVideoBench — Travel",
    channel: "LongVideoBench",
    youtubeId: "kOZnpwI2hIM",
    duration: "16:42",
  },
  lvb_history: {
    path: "longvideobench/videos/fvCrE5NCsts.mp4",
    title: "LongVideoBench — History",
    channel: "LongVideoBench",
    youtubeId: "fvCrE5NCsts",
    duration: "08:30",
  },
  lvb_art: {
    path: "longvideobench/videos/fZBC3nmvJb8.mp4",
    title: "LongVideoBench — Art",
    channel: "LongVideoBench",
    youtubeId: "fZBC3nmvJb8",
    duration: "20:06",
  },
  lvb_geography: {
    path: "longvideobench/videos/lzAESaVqix0.mp4",
    title: "LongVideoBench — Geography",
    channel: "LongVideoBench",
    youtubeId: "lzAESaVqix0",
    duration: "19:48",
  },
  lvb_stem: {
    path: "longvideobench/videos/zda-T6wrEhs.mp4",
    title: "LongVideoBench — STEM",
    channel: "LongVideoBench",
    youtubeId: "zda-T6wrEhs",
    duration: "08:30",
  },
  lvb_vlog: {
    path: "longvideobench/videos/Jfp1Ks7Hh1E.mp4",
    title: "LongVideoBench — Life Vlog",
    channel: "LongVideoBench",
    youtubeId: "Jfp1Ks7Hh1E",
    duration: "15:12",
  },
  lvb_napoleon: {
    path: "longvideobench/videos/P9hDA0u6FO0.mp4",
    title: "LongVideoBench — Napoleon",
    channel: "LongVideoBench",
    youtubeId: "P9hDA0u6FO0",
    duration: "33:18",
  },
  lvb_dejavu: {
    path: "longvideobench/videos/86CxyhFV9MI.mp4",
    title: "LongVideoBench — Deja Vu Stage",
    channel: "LongVideoBench",
    youtubeId: "86CxyhFV9MI",
    duration: "03:10",
  },
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

const LVB_DEJAVU_VIDEO_ID = "86CxyhFV9MI";
const LVB_DEJAVU_VIDEO_KEY = "lvb_dejavu";
const LVB_DEJAVU_PROMPTS = new Map<string, { id: number; name: string; question?: string }>([
  ["whats the summary of the video", { id: 22, name: "lvb_dejavu_summary" }],
  ["whats happening with this video", { id: 23, name: "lvb_dejavu_happening" }],
  [
    "can you give me a summary of what the video is about in 200 words and find how many people are dancing in the video",
    {
      id: 24,
      name: "lvb_dejavu_summary_dancers",
      question: "Give me a summary and how many dancers are present in the video",
    },
  ],
]);

function decodeQuotedString(raw: string): string {
  try {
    return JSON.parse(`"${raw}"`) as string;
  } catch {
    return raw
      .replace(/\\n/g, "\n")
      .replace(/\\"/g, '"')
      .replace(/\\'/g, "'");
  }
}

function normalizeQuestionKey(question: string): string {
  return question
    .toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function extractQuestionFromTraceEvents(traceEvents: TraceEvent[]): string | null {
  let preview: string | null = null;

  for (const event of traceEvents) {
    const payload = (event.payload ?? {}) as Record<string, unknown>;
    if (!preview && typeof payload.question_preview === "string") {
      preview = payload.question_preview.trim();
    }

    const code = payload.code_content;
    if (typeof code !== "string") continue;

    const m = code.match(/list_candidate_windows\(\s*question="([\s\S]*?)"\s*,/);
    if (m) {
      return decodeQuotedString(m[1]).trim();
    }
  }

  return preview;
}

function extractFinalAnswerFromCode(code: string): string | null {
  const triple = code.match(/done\(\s*"""([\s\S]*?)"""\s*\)\s*$/);
  if (triple) return triple[1].trim();

  const direct = code.match(/done\(\s*"([\s\S]*?)"\s*\)\s*$/);
  if (direct) return decodeQuotedString(direct[1]).trim();

  const doneVar = code.match(/done\(\s*([A-Za-z_]\w*)\s*\)\s*$/);
  if (!doneVar) return null;

  const varName = doneVar[1];
  const assignment = code.match(new RegExp(`${varName}\\s*=\\s*\\(([\\s\\S]*?)\\)\\s*done\\(\\s*${varName}\\s*\\)\\s*$`));
  if (!assignment) return null;

  const pieces = Array.from(assignment[1].matchAll(/"((?:\\.|[^"\\])*)"/g)).map((m) => decodeQuotedString(m[1]));
  if (pieces.length === 0) return null;

  return pieces.join("").trim();
}

async function loadLvbDejavuArtifactPrompts(): Promise<Array<{ version: string; data: ReturnType<typeof toCamelCase> }>> {
  const artifactsRoot = join(DATA_DIR, "longvideobench", "artifacts");

  type Candidate = {
    runDir: string;
    question: string;
    answerText: string;
    iterations: number;
    inputTokens: number;
    outputTokens: number;
    wallTimeS: number;
    evidenceCount: number;
    traceEvents: TraceEvent[];
  };

  let entries: Array<{ name: string; isDirectory: () => boolean }>;
  try {
    entries = await readdir(artifactsRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  const byQuestion = new Map<string, Candidate>();
  const runDirs = entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort()
    .reverse();

  for (const runDir of runDirs) {
    const manifestPath = join(artifactsRoot, runDir, "manifest.json");

    let manifest: { trace_events?: TraceEvent[] };
    try {
      manifest = JSON.parse(await readFile(manifestPath, "utf-8")) as { trace_events?: TraceEvent[] };
    } catch {
      continue;
    }

    const traceEvents = manifest.trace_events ?? [];
    if (!Array.isArray(traceEvents) || traceEvents.length === 0) continue;

    const runStart = traceEvents.find((e) => typeof (e.payload as Record<string, unknown> | undefined)?.video_path === "string");
    const videoPath = (runStart?.payload as Record<string, unknown> | undefined)?.video_path;
    if (typeof videoPath !== "string" || !videoPath.includes(`${LVB_DEJAVU_VIDEO_ID}.mp4`)) continue;

    const question = extractQuestionFromTraceEvents(traceEvents);
    if (!question) continue;

    let answerText = "";
    for (let i = traceEvents.length - 1; i >= 0; i--) {
      const payload = (traceEvents[i]?.payload ?? {}) as Record<string, unknown>;
      if (payload.has_final_answer && typeof payload.code_content === "string") {
        const extracted = extractFinalAnswerFromCode(payload.code_content);
        if (extracted) {
          answerText = extracted;
          break;
        }
      }
    }

    const iterations = traceEvents.reduce((max, e) => {
      const value = Number((e.payload as Record<string, unknown> | undefined)?.iteration);
      return Number.isFinite(value) ? Math.max(max, value) : max;
    }, 0);

    let inputTokens = 0;
    let outputTokens = 0;
    const evidenceIds = new Set<string>();
    const timestamps: number[] = [];

    for (const e of traceEvents) {
      const payload = (e.payload ?? {}) as Record<string, unknown>;
      inputTokens += Number(payload.input_tokens) || 0;
      outputTokens += Number(payload.output_tokens) || 0;

      if (typeof payload.clip_id === "string") {
        evidenceIds.add(payload.clip_id);
      }

      if (Number.isFinite(e.timestamp)) {
        timestamps.push(Number(e.timestamp));
      }
    }

    const wallTimeS = timestamps.length > 1 ? Math.max(0, timestamps[timestamps.length - 1] - timestamps[0]) : 0;
    const key = normalizeQuestionKey(question);

    const existing = byQuestion.get(key);
    const candidate: Candidate = {
      runDir,
      question,
      answerText,
      iterations,
      inputTokens,
      outputTokens,
      wallTimeS,
      evidenceCount: evidenceIds.size,
      traceEvents,
    };

    if (!existing || (!existing.answerText && candidate.answerText)) {
      byQuestion.set(key, candidate);
    }
  }

  let nextPromptId = 24;
  const results: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];

  for (const [questionKey, candidate] of Array.from(byQuestion.entries()).sort(([a], [b]) => a.localeCompare(b))) {
    const predefined = LVB_DEJAVU_PROMPTS.get(questionKey);
    const promptId = predefined?.id ?? nextPromptId++;
    const promptName = predefined?.name ?? `lvb_dejavu_prompt_${promptId}`;

    const raw: RawResult = {
      prompt_id: promptId,
      prompt_name: promptName,
      video_key: LVB_DEJAVU_VIDEO_KEY,
      question: predefined?.question ?? candidate.question,
      answer_text: candidate.answerText,
      answer_data: null,
      iterations: candidate.iterations || 1,
      cost_usd: 0,
      input_tokens: candidate.inputTokens,
      output_tokens: candidate.outputTokens,
      wall_time_s: candidate.wallTimeS,
      evidence_count: candidate.evidenceCount,
      evidence_sources: [`artifact:${candidate.runDir}`],
      subtitle: {
        had_existing_subtitle: true,
        subtitle_generated: false,
        subtitle_source: "existing",
      },
      trace_events: candidate.traceEvents,
    };

    results.push({ version: "v1", data: toCamelCase(raw) });
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

async function loadLvbDejavuLivePrompts(): Promise<Array<{ version: string; data: ReturnType<typeof toCamelCase> }>> {
  const artifactsDir = join(process.cwd(), "..", "sanjaya_artifacts");

  let runDirs: string[];
  try {
    runDirs = await readdir(artifactsDir);
  } catch {
    return [];
  }

  const byQuestion = new Map<string, { runDir: string; trace: TraceJson }>();
  const sorted = [...runDirs].sort().reverse();

  for (const runDir of sorted) {
    const tracePath = join(artifactsDir, runDir, "trace.json");

    let trace: TraceJson;
    try {
      trace = JSON.parse(await readFile(tracePath, "utf-8")) as TraceJson;
    } catch {
      continue;
    }

    const videoPath = extractVideoPath(trace);
    if (!videoPath || !videoPath.includes(`${LVB_DEJAVU_VIDEO_ID}.mp4`)) continue;

    const key = normalizeQuestionKey(trace.question ?? "");
    if (!LVB_DEJAVU_PROMPTS.has(key)) continue;

    const existing = byQuestion.get(key);
    const hasAnswer = Boolean(trace.answer || trace.answer_data);
    const existingHasAnswer = Boolean(existing?.trace.answer || existing?.trace.answer_data);

    if (!existing || (!existingHasAnswer && hasAnswer)) {
      byQuestion.set(key, { runDir, trace });
    }
  }

  const results: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];

  for (const [questionKey, { runDir, trace }] of Array.from(byQuestion.entries()).sort(([a], [b]) => a.localeCompare(b))) {
    const predefined = LVB_DEJAVU_PROMPTS.get(questionKey);
    if (!predefined) continue;

    const answerText = typeof trace.answer === "string" ? trace.answer : JSON.stringify(trace.answer ?? "");

    const raw: RawResult = {
      prompt_id: predefined.id,
      prompt_name: predefined.name,
      video_key: LVB_DEJAVU_VIDEO_KEY,
      question: predefined.question ?? trace.question,
      answer_text: answerText,
      answer_data: trace.answer_data ?? null,
      iterations: trace.iterations ?? 0,
      cost_usd: trace.cost?.total_cost_usd ?? 0,
      input_tokens: trace.cost?.total_input_tokens ?? 0,
      output_tokens: trace.cost?.total_output_tokens ?? 0,
      wall_time_s: trace.wall_time_s ?? 0,
      evidence_count: trace.evidence_count ?? 0,
      evidence_sources: [`artifact:${runDir}`],
      subtitle: {
        had_existing_subtitle: true,
        subtitle_generated: false,
        subtitle_source: "existing",
      },
      trace_events: trace.events,
    };

    results.push({ version: "v1", data: toCamelCase(raw) });
  }

  return results;
}

interface LiveRunEntry {
  runId: string;
  timestamp: string; // directory name e.g. "20260408-165850"
  model: string;
  videoPath: string | null;
  data: ReturnType<typeof toCamelCase>;
}

function shouldSkipVideoLiveHistory(trace: TraceJson, runDir: string): boolean {
  const runId = (trace.run_id ?? runDir).toLowerCase();

  if (runId.startsWith("live_run_docs")) return true;
  if (runId.startsWith("image_")) return true;

  // Keep explicitly tagged video runs.
  if (runId.startsWith("live_run_videos")) return false;

  const question = (trace.question ?? "").toLowerCase();
  if (question.includes("photo album retrieval")) return true;

  return false;
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
    if (runDir.toLowerCase().startsWith("image_")) continue;

    const tracePath = join(artifactsDir, runDir, "trace.json");
    const manifestPath = join(artifactsDir, runDir, "manifest.json");
    try {
      const trace = JSON.parse(await readFile(tracePath, "utf-8")) as TraceJson;
      if (shouldSkipVideoLiveHistory(trace, runDir)) continue;

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
  const lvbVersionDirs = await discoverLvbVersionDirs();

  // Load benchmark versions
  const allResults: Array<{ version: string; data: ReturnType<typeof toCamelCase> }> = [];
  for (const [version, dirName] of Object.entries(versionDirs)) {
    const results = await loadVersion(version, dirName);
    allResults.push(...results);
  }

  // Load LVB prompt versions from dedicated runs.
  for (const [version, dirName] of Object.entries(lvbVersionDirs)) {
    const results = await loadVersion(version, dirName);
    allResults.push(
      ...results.filter(({ data }) => data.videoKey.startsWith("lvb_") || data.promptId >= 13),
    );
  }

  const lvbDejavuResults = await loadLvbDejavuArtifactPrompts();
  allResults.push(...lvbDejavuResults);

  const lvbDejavuLiveResults = await loadLvbDejavuLivePrompts();
  allResults.push(...lvbDejavuLiveResults);

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
        videoPath: VIDEO_META[best.videoKey]?.path ?? null,
        question: best.question,
        versions,
        bestVersion,
        groundTruth: GROUND_TRUTH[promptId] ?? null,
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
    videos: Object.entries(VIDEO_META).map(([key, meta]) => ({
      key,
      path: meta.path,
      title: meta.title,
      channel: meta.channel,
      youtubeUrl: `https://www.youtube.com/watch?v=${meta.youtubeId}`,
      duration: meta.duration,
    })),
  });
}
