/** Types mirroring backend models and derived UI state. */

export interface TraceEvent {
  kind: string;
  timestamp: number;
  payload: Record<string, unknown>;
}

export type RunStatus = "idle" | "running" | "complete" | "error";

export interface RunState {
  runId: string | null;
  status: RunStatus;
  events: TraceEvent[];
  startTime: number | null;
  currentIteration: number;
  maxIterations: number;
  finalAnswer: string | null;
  finalStatus: string | null;
  error: string | null;
  orchestratorModel: string | null;
  recursiveModel: string | null;
}

/** Derived token/cost totals. */
export interface TokenTotals {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  costUsd: number;
}

/** A code execution entry. */
export interface CodeExecution {
  iteration: number;
  codeBlockIndex: number;
  codeBlockTotal: number;
  code: string;
  executionTime: number;
  stderr: string;
  hasFinalAnswer: boolean;
}

/** A sub-LLM call entry. */
export interface SubLLMCall {
  promptPreview: string;
  responsePreview: string;
  inputTokens: number | null;
  outputTokens: number | null;
  costUsd: number | null;
  modelUsed: string | null;
  durationSeconds: number | null;
}

/** Clip data from trace events. */
export interface ClipEntry {
  clipId: string;
  startS: number;
  endS: number;
  clipPath: string;
  frameCount: number;
}

/** Vision query entry. */
export interface VisionEntry {
  prompt: string;
  frameCount: number;
  clipCount: number;
  responsePreview: string;
  inputTokens: number | null;
  outputTokens: number | null;
  costUsd: number | null;
  modelUsed: string | null;
  durationSeconds: number | null;
  framePaths: string[];
  clipId: string | null;
}

/** Evidence item from final answer. */
export interface EvidenceItem {
  windowId: string | null;
  startS: number;
  endS: number;
  rationale: string;
}

/** A single prompt result from one version. */
export interface PromptResult {
  promptId: number;
  promptName: string;
  videoKey: string;
  question: string;
  answerText: string;
  answerData: Record<string, unknown> | null;
  iterations: number;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  wallTimeS: number;
  evidenceCount: number;
  evidenceSources: string[];
  subtitle: {
    hadExistingSubtitle: boolean;
    subtitleGenerated: boolean;
    subtitleSource: string;
  };
  error?: string;
  traceEvents?: TraceEvent[] | null;
}

/** Merged prompt with all available versions. */
export interface BenchmarkPrompt {
  promptId: number;
  promptName: string;
  videoKey: string;
  videoPath: string | null;
  question: string;
  versions: Record<string, PromptResult>;
  bestVersion: string;
  groundTruth?: string | null;
}

export interface LiveRunItem {
  runId: string;
  timestamp: string;
  model: string;
  prompt: BenchmarkPrompt;
}

export interface LiveRunsData {
  runs: LiveRunItem[];
  totalRuns: number;
  totalCostUsd: number;
  totalWallTimeS: number;
}

export interface VideoInfo {
  key: string;
  path: string;
  title: string;
  channel: string;
  youtubeUrl: string;
  duration: string;
}

export interface BenchmarkData {
  prompts: BenchmarkPrompt[];
  summary: {
    totalPrompts: number;
    versions: string[];
    totalCostUsd: number;
    totalWallTimeS: number;
    v1CostUsd: number;
    v1WallTimeS: number;
    latestVersion: string;
  };
  liveRuns: LiveRunsData;
  videos: VideoInfo[];
}

/* ── Document benchmark types ─────────────────────────── */

export interface DocumentResult {
  promptId: number;
  promptName: string;
  collection: string;
  documentPaths: string[];
  question: string;
  answerText: string;
  answerData: Record<string, unknown> | null;
  iterations: number;
  costUsd: number;
  inputTokens: number;
  outputTokens: number;
  wallTimeS: number;
  evidenceCount: number;
  evidenceSources: string[];
  error?: string;
  traceEvents?: TraceEvent[] | null;
}

export interface DocumentPrompt {
  promptId: number;
  promptName: string;
  collection: string;
  question: string;
  versions: Record<string, DocumentResult>;
  bestVersion: string;
}

export interface DocumentSource {
  name: string;
  type: string;
}

export interface DocumentBenchmarkData {
  prompts: DocumentPrompt[];
  summary: {
    totalPrompts: number;
    versions: string[];
    totalCostUsd: number;
    totalWallTimeS: number;
    latestVersion: string;
  };
  documents: DocumentSource[];
}

/* ── Image benchmark types ────────────────────────────── */

export interface MuirbenchChoice {
  label: string;
  text: string;
  isImageRef: boolean;
  imagePath: string | null;
  imageTag: string | null;
}

export interface MuirbenchMeta {
  idx: string;
  answer: string | null;
  choices: MuirbenchChoice[];
}

export interface ImageResult {
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
  error?: string;
  traceEvents?: TraceEvent[] | null;
  muirbench?: MuirbenchMeta | null;
}

export interface ImagePrompt {
  promptId: number;
  promptName: string;
  question: string;
  versions: Record<string, ImageResult>;
  bestVersion: string;
}

export interface BenchmarkAccuracy {
  correct: number;
  total: number;
  accuracy: number;
}

export interface ImageBenchmarkData {
  prompts: ImagePrompt[];
  summary: {
    totalPrompts: number;
    versions: string[];
    totalCostUsd: number;
    totalWallTimeS: number;
    latestVersion: string;
    models?: string[];
    benchmarks?: string[];
    benchmarkSummaries?: Record<string, BenchmarkAccuracy>;
  };
}
