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
}

/** Evidence item from final answer. */
export interface EvidenceItem {
  windowId: string | null;
  startS: number;
  endS: number;
  rationale: string;
}
