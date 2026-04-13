"use client";

import type { TraceEvent, RunStatus } from "@/lib/types";

interface FinalAnswerPanelProps {
  status: RunStatus;
  finalAnswer: string | null;
  finalStatus: string | null;
  question: string | null;
  events: TraceEvent[];
  iterations: number;
  costUsd: number;
  wallTimeS: number;
}

function deriveEvidence(events: TraceEvent[]) {
  // Extract evidence from clip events (windows that were investigated)
  const evidence: Array<{
    clipId: string;
    startS: number;
    endS: number;
    visionSummary: string | null;
  }> = [];

  const clipMap = new Map<string, { startS: number; endS: number }>();
  for (const e of events) {
    if (e.kind === "clip") {
      clipMap.set(e.payload.clip_id as string, {
        startS: (e.payload.start_s as number) ?? 0,
        endS: (e.payload.end_s as number) ?? 0,
      });
    }
  }

  // Match vision queries to clips
  const visionEvents = events.filter((e) => e.kind === "vision");
  for (const [clipId, clip] of clipMap) {
    const vision = visionEvents.find(
      (v) => ((v.payload.prompt as string) ?? "").includes(clip.startS.toString())
    );
    evidence.push({
      clipId,
      startS: clip.startS,
      endS: clip.endS,
      visionSummary: vision
        ? (vision.payload.response_preview as string) ?? null
        : null,
    });
  }

  return evidence;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function FinalAnswerPanel({
  status,
  finalAnswer,
  finalStatus,
  question,
  events,
  iterations,
  costUsd,
  wallTimeS,
}: FinalAnswerPanelProps) {
  // Only show when complete or error
  if (status !== "complete" && status !== "error") return null;

  const evidence = deriveEvidence(events);
  const isFinal = finalStatus === "final_answer";
  const isForced = finalStatus === "forced_final_answer";
  const isError = status === "error";

  return (
    <div className="border border-hud-border bg-hud-panel">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-hud-border px-4 py-2">
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 ${
              isError ? "bg-hud-red" : "bg-hud-green"
            }`}
          />
          <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-label">
            {isError ? "ERROR" : "FINAL ANSWER"}
          </span>
        </div>
        {!isError && (
          <span
            className={`text-[13px] uppercase tracking-wider px-2 py-0.5 border ${
              isFinal
                ? "border-hud-green text-hud-green"
                : isForced
                  ? "border-hud-amber text-hud-amber"
                  : "border-hud-dim text-hud-dim"
            }`}
          >
            {isFinal ? "FINAL ANSWER" : isForced ? "FORCED ANSWER" : finalStatus}
          </span>
        )}
      </div>

      <div className="p-4 space-y-4">
        {/* Stats row */}
        {!isError && (
          <div className="flex gap-6 text-[12px]">
            <div>
              <span className="text-hud-dim uppercase tracking-wider">Iterations </span>
              <span className="text-foreground font-bold tabular-nums">{iterations}</span>
            </div>
            <div>
              <span className="text-hud-dim uppercase tracking-wider">Time </span>
              <span className="text-foreground font-bold tabular-nums">{formatDuration(wallTimeS)}</span>
            </div>
            <div>
              <span className="text-hud-dim uppercase tracking-wider">Cost </span>
              <span className="text-hud-amber font-bold tabular-nums">${costUsd.toFixed(4)}</span>
            </div>
          </div>
        )}

        {/* Question */}
        {question && (
          <div>
            <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
              QUESTION
            </span>
            <p className="text-xs text-hud-label">{question}</p>
          </div>
        )}

        {/* Answer or Error */}
        {isError ? (
          <div className="border border-hud-red/30 bg-hud-red/5 p-3">
            <p className="text-sm text-hud-red">
              {finalAnswer ?? "Unknown error occurred"}
            </p>
          </div>
        ) : (
          <div>
            <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
              ANSWER
            </span>
            <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap break-words">
              {finalAnswer ?? "No answer available"}
            </p>
          </div>
        )}

        {/* Evidence */}
        {evidence.length > 0 && !isError && (
          <div>
            <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">
              EVIDENCE ({evidence.length})
            </span>
            <div className="space-y-0">
              {/* Header */}
              <div className="grid grid-cols-[1fr_60px_60px_2fr] gap-2 text-[13px] text-hud-dim uppercase tracking-wider border-b border-hud-border pb-1 mb-1">
                <span>WINDOW</span>
                <span className="text-right">START</span>
                <span className="text-right">END</span>
                <span>SUMMARY</span>
              </div>
              {evidence.map((e) => (
                <div
                  key={e.clipId}
                  className="grid grid-cols-[1fr_60px_60px_2fr] gap-2 text-[12px] py-0.5"
                >
                  <span className="text-foreground truncate">{e.clipId}</span>
                  <span className="text-right tabular-nums text-hud-dim">
                    {e.startS.toFixed(1)}s
                  </span>
                  <span className="text-right tabular-nums text-hud-dim">
                    {e.endS.toFixed(1)}s
                  </span>
                  <span className="text-hud-dim truncate">
                    {e.visionSummary ?? "—"}
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
