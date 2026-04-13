"use client";

import { cn } from "@/lib/utils";

export type RunStatus = "idle" | "running" | "complete" | "error";

interface StatusBarProps {
  status: RunStatus;
  runId: string | null;
  iteration: number;
  maxIterations: number;
  elapsedMs: number;
}

const statusConfig: Record<
  RunStatus,
  { label: string; color: string; dot: string }
> = {
  idle: { label: "IDLE", color: "text-hud-dim", dot: "bg-hud-dim" },
  running: {
    label: "RUNNING",
    color: "text-hud-amber",
    dot: "bg-hud-amber animate-pulse",
  },
  complete: {
    label: "COMPLETE",
    color: "text-hud-green",
    dot: "bg-hud-green",
  },
  error: { label: "ERROR", color: "text-hud-red", dot: "bg-hud-red" },
};

function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [
    String(hours).padStart(2, "0"),
    String(minutes).padStart(2, "0"),
    String(seconds).padStart(2, "0"),
  ].join(":");
}

export function StatusBar({
  status,
  runId,
  iteration,
  maxIterations,
  elapsedMs,
}: StatusBarProps) {
  const config = statusConfig[status];

  return (
    <div className="flex items-center justify-between border-b border-hud-border bg-hud-panel px-4 py-2">
      {/* Left: Status */}
      <div className="flex items-center gap-3">
        <span className={cn("h-2 w-2 shrink-0", config.dot)} />
        <span
          className={cn(
            "text-xs font-bold uppercase tracking-[0.2em]",
            config.color
          )}
        >
          {config.label}
        </span>
      </div>

      {/* Center: Iteration */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <span className="text-[12px] uppercase tracking-wider text-hud-dim">
            ITERATION
          </span>
          <span className="text-xs font-bold tabular-nums">
            {iteration} / {maxIterations}
          </span>
        </div>
        <div className="h-3 w-px bg-hud-border" />
        <div className="flex items-center gap-2">
          <span className="text-[12px] uppercase tracking-wider text-hud-dim">
            ELAPSED
          </span>
          <span className="text-xs font-bold tabular-nums">
            {formatElapsed(elapsedMs)}
          </span>
        </div>
      </div>

      {/* Right: Run ID */}
      <div className="flex items-center gap-2">
        <span className="text-[12px] uppercase tracking-wider text-hud-dim">
          RUN
        </span>
        <span className="text-xs tabular-nums text-hud-dim">
          {runId ?? "—"}
        </span>
      </div>
    </div>
  );
}
