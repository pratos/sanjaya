"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useRun } from "@/lib/use-run";
import { videoStreamUrl } from "@/lib/api";
import { useEffect, useRef, useState } from "react";

import { RunStatusHeader } from "@/components/hud/run-status";
import { QueryInput } from "@/components/hud/query-input";

import { TraceTimeline } from "@/components/hud/trace-timeline";

import { FinalAnswerPanel } from "@/components/hud/final-answer-panel";

interface RunHistoryEntry {
  runId: string;
  question: string;
  status: string;
  iterations: number;
  timestamp: number;
}

export default function RunPage() {
  const {
    state,
    startRun,
    reset,
    tokenTotals,
  } = useRun();

  const [question, setQuestion] = useState<string | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [history, setHistory] = useState<RunHistoryEntry[]>([]);
  const prevStatusRef = useRef(state.status);

  const isRunning = state.status === "running";

  // Compute wall time from events
  const wallTimeS = (() => {
    if (!state.startTime) return 0;
    const lastEvent = state.events[state.events.length - 1];
    const endTime = lastEvent?.timestamp ?? state.startTime;
    return endTime - state.startTime;
  })();

  // Auto-add to history when a run finishes
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = state.status;

    if (
      prev === "running" &&
      (state.status === "complete" || state.status === "error") &&
      state.runId
    ) {
      setHistory((h) => [
        {
          runId: state.runId!,
          question: question ?? "—",
          status: state.status,
          iterations: state.currentIteration,
          timestamp: Date.now(),
        },
        ...h,
      ]);
    }
  }, [state.status, state.runId, state.currentIteration, question]);

  return (
    <div className="flex min-h-screen flex-col">
      {/* Top bar */}
      <div className="border-b border-hud-border px-4 py-2 flex items-center gap-4">
        <Link
          href="/"
          className="flex items-center gap-1.5 text-xs text-hud-dim hover:text-foreground transition-colors"
        >
          <ArrowLeft size={14} />
          <span className="uppercase tracking-[0.15em]">Benchmarks</span>
        </Link>
        <div className="h-3 w-px bg-hud-border" />
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-foreground">
          Live Run
        </span>
        {state.status !== "idle" && (
          <button
            onClick={reset}
            disabled={isRunning}
            className="ml-auto text-[10px] uppercase tracking-[0.15em] border border-hud-border px-2 py-1 text-hud-dim hover:text-foreground hover:border-hud-border-active transition-colors disabled:opacity-30"
          >
            Reset
          </button>
        )}
      </div>

      {/* Status bar */}
      <RunStatusHeader
        status={state.status}
        runId={state.runId}
        iteration={state.currentIteration}
        maxIterations={state.maxIterations}
        startTime={state.startTime}
      />

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Query input */}
        <QueryInput
          onSubmit={(params) => {
            setQuestion(params.question);
            setVideoPath(params.videoPath);
            startRun(params);
          }}
          onVideoChange={setVideoPath}
          disabled={isRunning}
        />

        {/* Video preview */}
        {videoPath && (
          <div className="border border-hud-border bg-hud-panel">
            <div className="flex items-center gap-2 border-b border-hud-border px-3 py-2">
              <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-hud-label">
                Video
              </span>
              <span className="text-[9px] text-hud-dim truncate">{videoPath}</span>
            </div>
            <div className="p-3">
              <video
                src={videoStreamUrl(videoPath)}
                controls
                className="w-full max-h-64 bg-black"
              />
            </div>
          </div>
        )}

        {/* Trace timeline */}
        <div className="h-72 min-h-[18rem]">
          <TraceTimeline
            events={state.events}
            startTime={state.startTime}
          />
        </div>

        {/* Final answer */}
        <FinalAnswerPanel
          status={state.status}
          finalAnswer={state.finalAnswer}
          finalStatus={state.finalStatus}
          question={question}
          events={state.events}
          iterations={state.currentIteration}
          costUsd={tokenTotals.costUsd}
          wallTimeS={wallTimeS}
        />

        {/* Session history */}
        {history.length > 0 && (
          <div className="border border-hud-border bg-hud-panel">
            <div className="flex items-center gap-2 border-b border-hud-border px-3 py-2">
              <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-hud-label">
                Session History
              </span>
              <span className="text-[9px] text-hud-dim">
                {history.length} run{history.length !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="divide-y divide-hud-border/50">
              {history.map((entry) => (
                <div
                  key={entry.runId}
                  className="flex items-center gap-3 px-3 py-2 text-[10px]"
                >
                  <span
                    className={`h-1.5 w-1.5 shrink-0 ${
                      entry.status === "complete" ? "bg-hud-green" : "bg-hud-red"
                    }`}
                  />
                  <span className="font-mono text-hud-dim w-[7ch] shrink-0">
                    {entry.runId.slice(-6)}
                  </span>
                  <span className="truncate flex-1 text-foreground/80">
                    {entry.question}
                  </span>
                  <span className="text-hud-dim tabular-nums shrink-0">
                    {entry.iterations} iter
                  </span>
                  <span className="text-hud-dim shrink-0">
                    {new Date(entry.timestamp).toLocaleTimeString("en-GB", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
