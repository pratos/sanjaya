"use client";

import { useCallback, useRef, useState } from "react";
import { submitImageRun, streamEvents } from "./api";
import type { RunState, TraceEvent, TokenTotals } from "./types";

const initialState: RunState = {
  runId: null,
  status: "idle",
  events: [],
  startTime: null,
  currentIteration: 0,
  maxIterations: 10,
  finalAnswer: null,
  finalStatus: null,
  error: null,
  orchestratorModel: null,
  recursiveModel: null,
};

export function useImageRun() {
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
          updates.maxIterations = (p.max_iterations as number) ?? prev.maxIterations;
          updates.orchestratorModel = (p.orchestrator_model as string) ?? (p.model as string) ?? null;
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
          updates.currentIteration = (p.iteration as number) ?? prev.currentIteration;
          if (p.model) updates.orchestratorModel = p.model as string;
          break;
        case "run_end":
          updates.status = "complete";
          updates.finalAnswer =
            (p.final_answer as string) ?? (p.answer_full as string) ??
            (p.answer_preview as string) ?? prev.finalAnswer;
          updates.finalStatus = p.forced_answer ? "forced_final_answer" : prev.finalStatus ?? "final_answer";
          break;
      }

      return { ...prev, ...updates };
    });
  }, []);

  const startRun = useCallback(
    async (params: {
      imagePaths: string[];
      question: string;
      maxIterations?: number;
    }) => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }

      setState({ ...initialState, status: "running" });

      try {
        const { run_id } = await submitImageRun({
          image_paths: params.imagePaths,
          question: params.question,
          max_iterations: params.maxIterations,
        });

        setState((prev) => ({ ...prev, runId: run_id }));

        const cleanup = streamEvents(
          run_id,
          handleEvent,
          (error) => setState((prev) => ({ ...prev, status: "error", error })),
          () => {}
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

  const tokenTotals: TokenTotals = deriveTokenTotals(state.events);

  return { state, startRun, reset, tokenTotals };
}

function deriveTokenTotals(events: TraceEvent[]): TokenTotals {
  let inputTokens = 0, outputTokens = 0, totalTokens = 0, costUsd = 0;
  const kinds = new Set(["root_response", "sub_llm", "vision", "schema_generation", "critic_evaluation"]);
  for (const e of events) {
    if (kinds.has(e.kind)) {
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
