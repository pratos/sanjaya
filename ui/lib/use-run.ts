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
            (p.orchestrator_model as string) ?? (p.model as string) ?? null;
          updates.recursiveModel = (p.recursive_model as string) ?? null;
          break;

        case "iteration_start":
        case "iteration_end":
          updates.currentIteration = (p.iteration as number) ?? prev.currentIteration;
          if (p.final_answer) {
            updates.finalAnswer = p.final_answer as string;
            updates.finalStatus = p.forced_answer ? "forced_final_answer" : "final_answer";
          }
          break;

        case "root_response":
          // root_llm_call_end — carries response_preview, tokens, etc.
          updates.currentIteration = (p.iteration as number) ?? prev.currentIteration;
          if (p.model) {
            updates.orchestratorModel = p.model as string;
          }
          break;

        case "run_end": {
          // sanjaya.completion_end — run finished successfully
          updates.status = "complete";
          updates.finalAnswer =
            (p.final_answer as string) ??
            (p.answer_full as string) ??
            (p.answer_preview as string) ??
            (p.response_preview as string) ??
            prev.finalAnswer;
          updates.finalStatus = prev.finalStatus ?? "final_answer";
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

  const tokenEventKinds = new Set([
    "root_response", "sub_llm", "vision",
    "schema_generation", "critic_evaluation",
  ]);

  for (const e of events) {
    if (tokenEventKinds.has(e.kind)) {
      const p = e.payload;
      const inp = (p.input_tokens as number) ?? 0;
      const out = (p.output_tokens as number) ?? 0;
      inputTokens += inp;
      outputTokens += out;
      totalTokens += (p.total_tokens as number) ?? (inp + out);
      costUsd += (p.cost_usd as number) ?? 0;
    }
  }

  return { inputTokens, outputTokens, totalTokens, costUsd };
}

function deriveCodeExecutions(events: TraceEvent[]): CodeExecution[] {
  return events
    .filter((e) => e.kind === "code_execution")
    .map((e) => {
      const p = e.payload;
      return {
        iteration: (p.iteration as number) ?? (p.block_index as number) ?? 0,
        codeBlockIndex: (p.code_block_index as number) ?? (p.block_index as number) ?? 0,
        codeBlockTotal: (p.code_block_total as number) ?? 0,
        code: (p.code as string) ?? (p.code_content as string) ?? (p.code_preview as string) ?? "",
        executionTime: (p.execution_time as number) ?? (p.execution_time_s as number) ?? 0,
        stderr: (p.stderr as string) ?? (p.stderr_preview as string) ?? "",
        hasFinalAnswer: (p.has_final_answer as boolean) ?? (p.final_answer != null),
      };
    });
}

function deriveSubLLMCalls(events: TraceEvent[]): SubLLMCall[] {
  return events
    .filter((e) => e.kind === "sub_llm")
    .map((e) => {
      const p = e.payload;
      return {
        promptPreview: (p.prompt_preview as string) ?? (p.prompt_content as string)?.slice(0, 200) ?? "",
        responsePreview: (p.response_preview as string) ?? "",
        inputTokens: (p.input_tokens as number) ?? null,
        outputTokens: (p.output_tokens as number) ?? null,
        costUsd: (p.cost_usd as number) ?? null,
        modelUsed: (p.model_used as string) ?? (p.model as string) ?? null,
        durationSeconds: (p.duration_seconds as number) ?? null,
      };
    });
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
    .map((e) => {
      const p = e.payload;
      return {
        prompt: (p.prompt as string) ?? (p.prompt_content as string)?.slice(0, 200) ?? "",
        frameCount: (p.frame_count as number) ?? (p.n_frames as number) ?? 0,
        clipCount: (p.clip_count as number) ?? 0,
        responsePreview: (p.response_preview as string) ?? "",
        inputTokens: (p.input_tokens as number) ?? null,
        outputTokens: (p.output_tokens as number) ?? null,
        costUsd: (p.cost_usd as number) ?? null,
        modelUsed: (p.model_used as string) ?? (p.model as string) ?? null,
        durationSeconds: (p.duration_seconds as number) ?? null,
        framePaths: (p.frame_paths as string[]) ?? [],
        clipId: (p.clip_id as string) ?? null,
      };
    });
}
