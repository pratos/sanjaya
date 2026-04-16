import { readdir, readFile, access, stat } from "fs/promises";
import { join } from "path";
import { NextResponse } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");
const BENCHMARK_RESULTS_DIR = join(DATA_DIR, "benchmark_results");
const MUIRBENCH_MANIFEST_PATH = join(DATA_DIR, "benchmarks", "muirbench", "manifest.json");
const PHOTOBENCH_MANIFEST_PATH = join(DATA_DIR, "benchmarks", "photobench", "manifest.json");

/* ── Shared helpers ─────────────────────────────────────── */

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

/* ── MUIRBench manifest ─────────────────────────────────── */

interface MuirbenchChoiceRaw {
  label: string;
  text: string;
  is_image_ref: boolean;
}

interface MuirbenchSampleRaw {
  idx: string | number;
  options?: string[];
  choices?: MuirbenchChoiceRaw[];
  answer?: string;
  image_paths?: Array<string | null>;
}

interface MuirbenchChoice {
  label: string;
  text: string;
  isImageRef: boolean;
  imagePath: string | null;
  imageTag: string | null;
}

interface MuirbenchMeta {
  idx: string;
  answer: string | null;
  choices: MuirbenchChoice[];
}

let muirbenchIndexPromise: Promise<Map<string, MuirbenchSampleRaw>> | null = null;

function toChoiceLabel(index: number): string {
  return String.fromCharCode("A".charCodeAt(0) + index);
}

function fileStem(path: string | null): string | null {
  if (!path) return null;
  const fileName = path.split("/").filter(Boolean).pop();
  if (!fileName) return null;
  return fileName.replace(/\.[^.]+$/, "");
}

async function loadMuirbenchIndex(): Promise<Map<string, MuirbenchSampleRaw>> {
  try {
    const raw = await readFile(MUIRBENCH_MANIFEST_PATH, "utf-8");
    const samples = JSON.parse(raw) as MuirbenchSampleRaw[];
    const index = new Map<string, MuirbenchSampleRaw>();
    for (const sample of samples) {
      index.set(String(sample.idx), sample);
    }
    return index;
  } catch {
    return new Map<string, MuirbenchSampleRaw>();
  }
}

function getMuirbenchIndex(): Promise<Map<string, MuirbenchSampleRaw>> {
  if (!muirbenchIndexPromise) {
    muirbenchIndexPromise = loadMuirbenchIndex();
  }
  return muirbenchIndexPromise;
}

/* ── PhotoBench manifest ────────────────────────────────── */

interface PhotobenchAlbumRaw {
  name: string;
  path: string;
  image_count: number;
}

interface PhotobenchQueryRaw {
  query_id: string;
  query: string;
  album: string;
  ground_truth: string[];
}

interface PhotobenchManifestRaw {
  albums?: PhotobenchAlbumRaw[];
  queries?: {
    train?: PhotobenchQueryRaw[];
    test?: PhotobenchQueryRaw[];
  };
}

interface PhotobenchQueryMeta {
  queryId: string;
  query: string;
  album: string;
  albumPath: string | null;
  groundTruth: string[];
}

interface PhotobenchManifestIndex {
  albumPathByName: Map<string, string>;
  expectedQueriesByAlbum: Map<string, number>;
  queryById: Map<string, PhotobenchQueryMeta>;
}

let photobenchManifestIndexPromise: Promise<PhotobenchManifestIndex> | null = null;

async function loadPhotobenchManifestIndex(): Promise<PhotobenchManifestIndex> {
  try {
    const raw = await readFile(PHOTOBENCH_MANIFEST_PATH, "utf-8");
    const manifest = JSON.parse(raw) as PhotobenchManifestRaw;

    const albumPathByName = new Map<string, string>();
    for (const album of manifest.albums ?? []) {
      if (album?.name && album?.path) {
        albumPathByName.set(album.name, album.path);
      }
    }

    const queryById = new Map<string, PhotobenchQueryMeta>();
    const expectedQueriesByAlbum = new Map<string, number>();
    const allQueries = [
      ...(manifest.queries?.test ?? []),
      ...(manifest.queries?.train ?? []),
    ];

    for (const query of allQueries) {
      const album = query.album;
      if (!album || !query.query_id) continue;

      const currentCount = expectedQueriesByAlbum.get(album) ?? 0;
      expectedQueriesByAlbum.set(album, currentCount + 1);

      queryById.set(query.query_id, {
        queryId: query.query_id,
        query: query.query,
        album,
        albumPath: albumPathByName.get(album) ?? null,
        groundTruth: Array.isArray(query.ground_truth) ? query.ground_truth : [],
      });
    }

    return {
      albumPathByName,
      expectedQueriesByAlbum,
      queryById,
    };
  } catch {
    return {
      albumPathByName: new Map<string, string>(),
      expectedQueriesByAlbum: new Map<string, number>(),
      queryById: new Map<string, PhotobenchQueryMeta>(),
    };
  }
}

function getPhotobenchManifestIndex(): Promise<PhotobenchManifestIndex> {
  if (!photobenchManifestIndexPromise) {
    photobenchManifestIndexPromise = loadPhotobenchManifestIndex();
  }
  return photobenchManifestIndexPromise;
}

/* ── Config loading ─────────────────────────────────────── */

interface RunConfig {
  benchmark: string;
  run_name: string;
  model?: string;
  vision_model?: string;
  caption_model?: string;
  max_iterations?: number;
  [key: string]: unknown;
}

async function loadConfig(dir: string): Promise<RunConfig | null> {
  try {
    const raw = await readFile(join(dir, "config.json"), "utf-8");
    return JSON.parse(raw) as RunConfig;
  } catch {
    return null;
  }
}

/* ── MUIRBench results ──────────────────────────────────── */

interface MuirbenchResultRaw {
  idx: string;
  task: string;
  image_relation?: string;
  question: string;
  answer_gt: string;
  predicted: string;
  correct: boolean;
  raw_answer: string;
  reasoning?: string;
  iterations: number;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  wall_time_s: number;
  n_images: number;
  trace_events?: TraceEvent[];
}

function muirbenchToImageResult(
  raw: MuirbenchResultRaw,
  promptId: number,
  muirbenchIndex: Map<string, MuirbenchSampleRaw>,
): NormalizedResult {
  const sample = muirbenchIndex.get(String(raw.idx));

  // Build MuirbenchMeta from manifest + result
  const sampleImagePaths: string[] = sample && Array.isArray(sample.image_paths)
    ? sample.image_paths.filter((p): p is string => typeof p === "string")
    : [];

  let muirbenchMeta: MuirbenchMeta | null = null;
  if (sample) {
    const choiceDefs =
      Array.isArray(sample.choices) && sample.choices.length > 0
        ? sample.choices
        : Array.isArray(sample.options)
          ? sample.options.map((opt, i) => ({
              label: toChoiceLabel(i),
              text: opt,
              is_image_ref: opt === "<image>",
            }))
          : [];

    const choices: MuirbenchChoice[] = choiceDefs.map((choice, optionIndex) => {
      const isImageRef = Boolean(choice.is_image_ref);
      const imagePath = isImageRef
        ? sampleImagePaths[optionIndex] ?? null
        : null;
      return {
        label: choice.label || toChoiceLabel(optionIndex),
        text: choice.text ?? "",
        isImageRef,
        imagePath,
        imageTag: fileStem(imagePath),
      };
    });

    muirbenchMeta = {
      idx: String(raw.idx),
      answer: typeof sample.answer === "string" ? sample.answer : null,
      choices,
    };
  }

  return {
    promptId,
    promptName: `muir_${raw.idx}_${(raw.task ?? "unknown").replace(/\s+/g, "_").toLowerCase()}`,
    imagePaths: sampleImagePaths,
    question: raw.question,
    answerText: raw.raw_answer ?? "",
    answerData: {
      benchmark: "MUIRBench",
      idx: raw.idx,
      task: raw.task,
      image_relation: raw.image_relation ?? null,
      predicted: raw.predicted,
      ground_truth: raw.answer_gt,
      correct: raw.correct,
    } as Record<string, unknown>,
    groundTruth: raw.answer_gt,
    iterations: raw.iterations ?? 0,
    costUsd: raw.cost_usd ?? 0,
    inputTokens: raw.input_tokens ?? 0,
    outputTokens: raw.output_tokens ?? 0,
    wallTimeS: raw.wall_time_s ?? 0,
    evidenceCount: raw.n_images ?? 0,
    evidenceSources: [] as string[],
    error: undefined as string | undefined,
    traceEvents: raw.trace_events ? normalizeTraceEvents(raw.trace_events) : null,
    muirbench: muirbenchMeta,
  };
}

/* ── PhotoBench results ─────────────────────────────────── */

interface PhotobenchResultRaw {
  query_id: string;
  query: string;
  predictions: string[];
  ground_truth: string[];
  metrics: {
    precision: number;
    recall: number;
    f1: number;
    mrr: number;
    tp: number;
    fp: number;
    fn: number;
  };
  reasoning?: string;
  evidence?: unknown[];
  raw_answer: string;
  iterations: number;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  wall_time_s: number;
  n_album_images: number;
  trace_events?: TraceEvent[];
}

function unique(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (!value) continue;
    if (seen.has(value)) continue;
    seen.add(value);
    result.push(value);
  }
  return result;
}

function photobenchToImageResult(
  raw: PhotobenchResultRaw,
  promptId: number,
  photobenchIndex: PhotobenchManifestIndex,
): NormalizedResult {
  const queryMeta = photobenchIndex.queryById.get(raw.query_id);
  const album = queryMeta?.album ?? raw.query_id.split("_")[0] ?? "unknown";
  const albumPath = queryMeta?.albumPath ?? photobenchIndex.albumPathByName.get(album) ?? null;

  const predictionImagePaths = unique(
    (raw.predictions ?? []).map((fileName) => (albumPath ? join(albumPath, "images", fileName) : null)),
  );

  const groundTruthValues = raw.ground_truth ?? queryMeta?.groundTruth ?? [];
  const groundTruthImagePaths = unique(
    groundTruthValues.map((fileName) => (albumPath ? join(albumPath, "images", fileName) : null)),
  );

  const imagePaths = unique([...groundTruthImagePaths, ...predictionImagePaths]);

  return {
    promptId,
    promptName: `photo_${raw.query_id}`,
    imagePaths,
    question: raw.query,
    answerText: raw.raw_answer ?? "",
    answerData: {
      benchmark: "PhotoBench",
      query_id: raw.query_id,
      album,
      album_path: albumPath,
      predictions: raw.predictions ?? [],
      ground_truth: groundTruthValues,
      prediction_image_paths: predictionImagePaths,
      ground_truth_image_paths: groundTruthImagePaths,
      metrics: raw.metrics,
      reasoning: raw.reasoning ?? null,
      evidence: raw.evidence ?? [],
      correct: raw.metrics?.tp > 0 && raw.metrics?.fp === 0 && raw.metrics?.fn === 0,
    } as Record<string, unknown>,
    groundTruth: groundTruthValues.join(", ") || null,
    iterations: raw.iterations ?? 0,
    costUsd: raw.cost_usd ?? 0,
    inputTokens: raw.input_tokens ?? 0,
    outputTokens: raw.output_tokens ?? 0,
    wallTimeS: raw.wall_time_s ?? 0,
    evidenceCount: raw.n_album_images ?? 0,
    evidenceSources: [] as string[],
    error: undefined as string | undefined,
    traceEvents: raw.trace_events ? normalizeTraceEvents(raw.trace_events) : null,
    muirbench: null,
  };
}

/* ── Checkpoint fallback (in-progress runs) ─────────────── */

function parseCheckpointLines(text: string): unknown[] {
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter((obj): obj is Record<string, unknown> => obj !== null);
}

/* ── Run discovery & loading ────────────────────────────── */

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

interface NormalizedResult {
  promptId: number;
  promptName: string;
  imagePaths: string[];
  question: string;
  answerText: string;
  answerData: Record<string, unknown> | null;
  groundTruth: string | null;
  iterations: number;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  wallTimeS: number;
  evidenceCount: number;
  evidenceSources: string[];
  error: string | undefined;
  traceEvents: ReturnType<typeof normalizeTraceEvents> | null;
  muirbench: MuirbenchMeta | null;
}

interface LoadedRun {
  benchmark: string;
  runName: string;
  config: RunConfig;
  results: NormalizedResult[];
  runDirName: string;
  updatedAtMs: number;
}

async function loadRun(
  benchmarkName: string,
  runDirName: string,
  runDir: string,
  muirbenchIndex: Map<string, MuirbenchSampleRaw>,
  photobenchIndex: PhotobenchManifestIndex,
): Promise<LoadedRun | null> {
  const config = await loadConfig(runDir);
  if (!config) return null;

  const runName = config.run_name ?? runDirName;
  const benchmark = config.benchmark?.toLowerCase() ?? benchmarkName;

  // Try results.json first, fall back to checkpoint.jsonl
  let rawResults: unknown[] = [];
  const resultsPath = join(runDir, "results.json");
  const checkpointPath = join(runDir, "checkpoint.jsonl");

  let sourcePath: string | null = null;
  if (await exists(resultsPath)) {
    try {
      const raw = await readFile(resultsPath, "utf-8");
      rawResults = JSON.parse(raw);
      sourcePath = resultsPath;
    } catch {
      // fall through to checkpoint
    }
  }

  if (rawResults.length === 0 && (await exists(checkpointPath))) {
    try {
      const raw = await readFile(checkpointPath, "utf-8");
      rawResults = parseCheckpointLines(raw);
      sourcePath = checkpointPath;
    } catch {
      // skip
    }
  }

  if (rawResults.length === 0) return null;

  let promptId = 1;
  const results = rawResults.map((raw) => {
    if (benchmark === "muirbench") {
      return muirbenchToImageResult(raw as MuirbenchResultRaw, promptId++, muirbenchIndex);
    }
    if (benchmark === "photobench") {
      return photobenchToImageResult(raw as PhotobenchResultRaw, promptId++, photobenchIndex);
    }
    // Generic fallback — treat as muirbench-like
    return muirbenchToImageResult(raw as MuirbenchResultRaw, promptId++, muirbenchIndex);
  });

  let updatedAtMs = 0;
  if (sourcePath) {
    try {
      updatedAtMs = (await stat(sourcePath)).mtimeMs;
    } catch {
      updatedAtMs = 0;
    }
  }

  return { benchmark, runName, config, results, runDirName, updatedAtMs };
}

function inferPhotobenchAlbum(run: LoadedRun): string | null {
  const albumsConfig = run.config.albums;
  if (albumsConfig && typeof albumsConfig === "object") {
    const keys = Object.keys(albumsConfig as Record<string, unknown>);
    if (keys.length > 0) return keys[0];
  }

  const first = run.results[0];
  const queryId = first?.answerData?.["query_id"];
  if (typeof queryId === "string" && queryId.includes("_")) {
    return queryId.split("_")[0] ?? null;
  }

  return null;
}

function selectLatestFullPhotobenchRuns(
  runs: LoadedRun[],
  photobenchIndex: PhotobenchManifestIndex,
): LoadedRun[] {
  const nonPhotobench = runs.filter((run) => run.benchmark !== "photobench");
  const photobenchRuns = runs.filter((run) => run.benchmark === "photobench");

  const latestByAlbum = new Map<string, LoadedRun>();

  for (const run of photobenchRuns) {
    const album = inferPhotobenchAlbum(run);
    if (!album) continue;

    const totalQueriesConfig = typeof run.config.total_queries === "number"
      ? run.config.total_queries
      : null;
    const expectedByManifest = photobenchIndex.expectedQueriesByAlbum.get(album) ?? null;
    const expected = totalQueriesConfig ?? expectedByManifest ?? 0;
    const isFull = expected > 0 ? run.results.length >= expected : run.results.length > 0;
    if (!isFull) continue;

    const existing = latestByAlbum.get(album);
    if (!existing) {
      latestByAlbum.set(album, run);
      continue;
    }

    const isNewer =
      run.updatedAtMs > existing.updatedAtMs ||
      (run.updatedAtMs === existing.updatedAtMs && run.runDirName > existing.runDirName);

    if (isNewer) {
      latestByAlbum.set(album, run);
    }
  }

  return [...nonPhotobench, ...Array.from(latestByAlbum.values())];
}

async function discoverRuns(): Promise<LoadedRun[]> {
  const muirbenchIndex = await getMuirbenchIndex();
  const photobenchIndex = await getPhotobenchManifestIndex();
  const runs: LoadedRun[] = [];

  let benchmarkDirs: string[];
  try {
    benchmarkDirs = await readdir(BENCHMARK_RESULTS_DIR);
  } catch {
    return runs;
  }

  for (const benchmarkName of benchmarkDirs.sort()) {
    const benchmarkDir = join(BENCHMARK_RESULTS_DIR, benchmarkName);

    let runDirs: string[];
    try {
      runDirs = await readdir(benchmarkDir);
    } catch {
      continue;
    }

    for (const runDirName of runDirs.sort()) {
      const runDir = join(benchmarkDir, runDirName);
      const run = await loadRun(benchmarkName, runDirName, runDir, muirbenchIndex, photobenchIndex);
      if (run) runs.push(run);
    }
  }

  const selectedRuns = selectLatestFullPhotobenchRuns(runs, photobenchIndex);
  return selectedRuns.sort((a, b) => a.benchmark.localeCompare(b.benchmark) || a.runName.localeCompare(b.runName));
}

/* ── GET handler ────────────────────────────────────────── */

export async function GET() {
  const runs = await discoverRuns();

  // Build version label for each run: "{benchmark}/{run_name}"
  const allResults: Array<{
    version: string;
    benchmark: string;
    model: string | null;
    data: NormalizedResult;
  }> = [];

  for (const run of runs) {
    const version = `${run.benchmark}/${run.runName}`;
    const model =
      run.config.model ?? run.config.vision_model ?? null;

    for (const result of run.results) {
      allResults.push({ version, benchmark: run.benchmark, model, data: result });
    }
  }

  // Group by promptName across versions
  const byName = new Map<
    string,
    Record<string, NormalizedResult>
  >();
  for (const { version, data } of allResults) {
    const key = data.promptName;
    if (!byName.has(key)) byName.set(key, {});
    byName.get(key)![version] = data;
  }

  let promptId = 1;
  const prompts = Array.from(byName.entries()).map(([name, versions]) => {
    const sortedVersions = Object.keys(versions).sort();
    let bestVersion = sortedVersions[0] ?? "v1";
    for (const v of sortedVersions) {
      if (versions[v] && !versions[v].error) {
        bestVersion = v;
        break;
      }
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
  let totalCostUsd = 0;
  let totalWallTimeS = 0;
  const versionSet = new Set<string>();
  const modelSet = new Set<string>();
  const benchmarkSet = new Set<string>();

  for (const { version, benchmark, model } of allResults) {
    versionSet.add(version);
    benchmarkSet.add(benchmark);
    if (model) modelSet.add(model);
  }

  for (const p of prompts) {
    const best = p.versions[p.bestVersion];
    if (best && !best.error) {
      totalCostUsd += best.costUsd;
      totalWallTimeS += best.wallTimeS;
    }
  }

  const sortedVersions = Array.from(versionSet).sort();

  // Per-benchmark accuracy summaries (pick run with most results)
  const benchmarkSummaries: Record<
    string,
    { correct: number; total: number; accuracy: number; version: string }
  > = {};
  for (const bm of benchmarkSet) {
    const bmVersions = sortedVersions.filter((v) => v.startsWith(bm + "/"));
    let bestVersion = bmVersions[0] ?? "";
    let bestCount = 0;
    for (const v of bmVersions) {
      const count = allResults.filter((r) => r.version === v).length;
      if (count > bestCount) {
        bestCount = count;
        bestVersion = v;
      }
    }
    const versionResults = allResults.filter((r) => r.version === bestVersion);
    const correct = versionResults.filter(
      (r) => (r.data.answerData as Record<string, unknown>)?.correct === true,
    ).length;
    benchmarkSummaries[bm] = {
      correct,
      total: versionResults.length,
      accuracy: versionResults.length > 0 ? correct / versionResults.length : 0,
      version: bestVersion,
    };
  }

  return NextResponse.json({
    prompts,
    summary: {
      totalPrompts: prompts.length,
      versions: sortedVersions,
      totalCostUsd: Math.round(totalCostUsd * 10000) / 10000,
      totalWallTimeS: Math.round(totalWallTimeS * 10) / 10,
      latestVersion: sortedVersions[sortedVersions.length - 1] ?? "v1",
      models: Array.from(modelSet),
      benchmarks: Array.from(benchmarkSet),
      benchmarkSummaries,
    },
  });
}
