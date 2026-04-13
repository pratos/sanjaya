"use client";

import { Panel } from "./panel";
import { DataRow } from "./data-row";
import type { RunStatus } from "@/lib/types";

interface ProgressBarProps {
  status: RunStatus;
  currentIteration: number;
  maxIterations: number;
  orchestratorModel: string | null;
  recursiveModel: string | null;
}

export function ProgressBar({
  status,
  currentIteration,
  maxIterations,
  orchestratorModel,
  recursiveModel,
}: ProgressBarProps) {
  const progressPct =
    maxIterations > 0 ? (currentIteration / maxIterations) * 100 : 0;

  // Barcode-style: render discrete stripes
  const totalStripes = maxIterations;
  const filledStripes = currentIteration;

  return (
    <Panel title="PROGRESS">
      <div className="flex flex-col justify-center gap-3 h-full">
        {/* Barcode progress bar */}
        <div className="flex items-center gap-2">
          <div className="flex-1 flex gap-[1px] h-4">
            {Array.from({ length: totalStripes }, (_, i) => (
              <div
                key={i}
                className={`flex-1 transition-all duration-300 ${
                  i < filledStripes
                    ? status === "error"
                      ? "bg-hud-red"
                      : status === "complete"
                        ? "bg-hud-green"
                        : "bg-hud-green/80"
                    : "bg-hud-border/50"
                }`}
              />
            ))}
          </div>
          <span className="text-[12px] tabular-nums text-hud-dim whitespace-nowrap">
            {currentIteration}/{maxIterations}
          </span>
        </div>

        {/* Percentage */}
        <div className="text-center">
          <span className="text-lg font-bold tabular-nums">
            {progressPct.toFixed(0)}%
          </span>
        </div>

        {/* Model info */}
        {orchestratorModel && (
          <DataRow label="MODEL" value={orchestratorModel} />
        )}
        {recursiveModel && (
          <DataRow label="SUB-MODEL" value={recursiveModel} />
        )}
        {status === "idle" && (
          <span className="text-[12px] uppercase tracking-wider text-hud-dim text-center">
            AWAITING INPUT
          </span>
        )}
        {status === "complete" && (
          <span className="text-[12px] uppercase tracking-wider text-hud-green text-center font-bold">
            COMPLETE
          </span>
        )}
        {status === "error" && (
          <span className="text-[12px] uppercase tracking-wider text-hud-red text-center font-bold">
            ERROR
          </span>
        )}
      </div>
    </Panel>
  );
}
