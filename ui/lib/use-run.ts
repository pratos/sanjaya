"use client";

import { useCallback, useRef, useState } from "react";
import { submitRun, streamEvents } from "./api";
import type {
  RunState,
  TraceEvent,
  TokenTotals,
  CodeExecution,
  SubLLMCall,
  ClipEntry,
  VisionEntry,
} from "./types";

const initialState: RunState = {
  runId: null,
  status: "idle",
  events: [],
  startTime: null,
  currentIteration: 0,
  maxIterations: 20,
  finalAnswer: null,
  finalStatus: null,
  error: null,
  orchestratorModel: null,
  recursiveModel: null,
};

export function useRun() {
  const [state, setState] = useState<RunState>(initialState);
  const cleanupRef = useRef<(() => void) | null>(null);

  const handleEvent = useCallback((event: TraceEvent) => {
    setState((prev) => {
      const events = [...prev.events, event];
      const p = event.payload;

      const updates: Partial<RunState> = { events };

      switch (event.kind) {
        case "run_start":
          updates.status = "running";
          updates.startTime = event.timestamp;
          updates.maxIterations =
            (p.max_iterations as number) ?? prev.maxIterations;
          updates.orchestratorModel =
            (p.orchestrator_model as string) ?? null;
          updates.recursiveModel = (p.recursive_model as string) ?? null;
          break;

        case "root_response":
          updates.currentIteration = (p.iteration as number) ?? prev.currentIteration;
          break;

        case "run_end": {
          const status = p.status as string;
          updates.status =
            status === "error" ? "error" : "complete";
          updates.finalAnswer = (p.answer_full as string) ?? (p.answer_preview as string) ?? null;
          updates.finalStatus = status;
          break;
        }

        default:
          break;
      }

      return { ...prev, ...updates };
    });
  }, []);

  const startRun = useCallback(
    async (params: {
      videoPath: string;
      question: string;
      subtitleMode?: string;
      subtitleApiModel?: string;
      maxIterations?: number;
    }) => {
      // Cleanup any existing connection
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }

      // Reset state
      setState({ ...initialState, status: "running" });

      try {
        const { run_id } = await submitRun({
          video_path: params.videoPath,
          question: params.question,
          subtitle_mode: params.subtitleMode,
          subtitle_api_model: params.subtitleApiModel,
          max_iterations: params.maxIterations,
        });

        setState((prev) => ({ ...prev, runId: run_id }));

        const cleanup = streamEvents(
          run_id,
          handleEvent,
          (error) => {
            setState((prev) => ({
              ...prev,
              status: "error",
              error,
            }));
          },
          () => {
            // stream ended — status already set by run_end event
          }
        );

        cleanupRef.current = cleanup;
      } catch (err) {
        setState((prev) => ({
          ...prev,
          status: "error",
          error: err instanceof Error ? err.message : String(err),
        }));
      }
    },
    [handleEvent]
  );

  const reset = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    setState(initialState);
  }, []);

  // Derived data
  const tokenTotals: TokenTotals = deriveTokenTotals(state.events);
  const codeExecutions: CodeExecution[] = deriveCodeExecutions(state.events);
  const subLLMCalls: SubLLMCall[] = deriveSubLLMCalls(state.events);
  const clips: ClipEntry[] = deriveClips(state.events);
  const visionQueries: VisionEntry[] = deriveVisionQueries(state.events);

  return {
    state,
    startRun,
    reset,
    tokenTotals,
    codeExecutions,
    subLLMCalls,
    clips,
    visionQueries,
  };
}

function deriveTokenTotals(events: TraceEvent[]): TokenTotals {
  let inputTokens = 0;
  let outputTokens = 0;
  let totalTokens = 0;
  let costUsd = 0;

  for (const e of events) {
    if (
      e.kind === "root_response" ||
      e.kind === "sub_llm" ||
      e.kind === "vision"
    ) {
      const p = e.payload;
      inputTokens += (p.input_tokens as number) ?? 0;
      outputTokens += (p.output_tokens as number) ?? 0;
      totalTokens += (p.total_tokens as number) ?? 0;
      costUsd += (p.cost_usd as number) ?? 0;
    }
  }

  return { inputTokens, outputTokens, totalTokens, costUsd };
}

function deriveCodeExecutions(events: TraceEvent[]): CodeExecution[] {
  return events
    .filter((e) => e.kind === "code_execution")
    .map((e) => ({
      iteration: (e.payload.iteration as number) ?? 0,
      codeBlockIndex: (e.payload.code_block_index as number) ?? 0,
      codeBlockTotal: (e.payload.code_block_total as number) ?? 0,
      code: (e.payload.code_content as string) ?? (e.payload.code_preview as string) ?? "",
      executionTime: (e.payload.execution_time as number) ?? 0,
      stderr: (e.payload.stderr_preview as string) ?? "",
      hasFinalAnswer: (e.payload.has_final_answer as boolean) ?? false,
    }));
}

function deriveSubLLMCalls(events: TraceEvent[]): SubLLMCall[] {
  return events
    .filter((e) => e.kind === "sub_llm")
    .map((e) => ({
      promptPreview: (e.payload.prompt_preview as string) ?? "",
      responsePreview: (e.payload.response_preview as string) ?? "",
      inputTokens: (e.payload.input_tokens as number) ?? null,
      outputTokens: (e.payload.output_tokens as number) ?? null,
      costUsd: (e.payload.cost_usd as number) ?? null,
      modelUsed: (e.payload.model_used as string) ?? null,
      durationSeconds: (e.payload.duration_seconds as number) ?? null,
    }));
}

function deriveClips(events: TraceEvent[]): ClipEntry[] {
  const clipMap = new Map<string, ClipEntry>();

  for (const e of events) {
    if (e.kind === "clip") {
      const clipId = e.payload.clip_id as string;
      clipMap.set(clipId, {
        clipId,
        startS: (e.payload.start_s as number) ?? 0,
        endS: (e.payload.end_s as number) ?? 0,
        clipPath: (e.payload.clip_path as string) ?? "",
        frameCount: 0,
      });
    }
    if (e.kind === "frames") {
      const clipId = e.payload.clip_id as string;
      const existing = clipMap.get(clipId);
      if (existing) {
        existing.frameCount = (e.payload.frame_count as number) ?? 0;
      }
    }
  }

  return Array.from(clipMap.values());
}

function deriveVisionQueries(events: TraceEvent[]): VisionEntry[] {
  return events
    .filter((e) => e.kind === "vision")
    .map((e) => ({
      prompt: (e.payload.prompt as string) ?? "",
      frameCount: (e.payload.frame_count as number) ?? 0,
      clipCount: (e.payload.clip_count as number) ?? 0,
      responsePreview: (e.payload.response_preview as string) ?? "",
      inputTokens: (e.payload.input_tokens as number) ?? null,
      outputTokens: (e.payload.output_tokens as number) ?? null,
      costUsd: (e.payload.cost_usd as number) ?? null,
      modelUsed: (e.payload.model_used as string) ?? null,
      durationSeconds: (e.payload.duration_seconds as number) ?? null,
    }));
}
