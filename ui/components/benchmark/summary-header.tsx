"use client";

import Link from "next/link";
import { Play } from "lucide-react";
import type { BenchmarkData } from "@/lib/types";

interface SummaryHeaderProps {
  data: BenchmarkData;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function SummaryHeader({ data }: SummaryHeaderProps) {
  return (
    <div className="border-b border-hud-border bg-hud-panel px-6 py-5">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-[0.3em] text-foreground lowercase">
            sanjaya
          </h1>
          <p className="text-[12px] lowercase tracking-[0.15em] text-hud-dim mt-1">
            video analysis dashboard
          </p>
        </div>
        <div className="flex items-center gap-8">
          <Stat label="PROMPTS" value={String(data.summary.totalPrompts)} />
          <Stat label="VERSIONS" value={data.summary.versions.join(", ")} />
          <Stat
            label="V1 COST"
            value={`$${data.summary.v1CostUsd.toFixed(2)}`}
          />
          <Stat
            label="V1 TIME"
            value={formatTime(data.summary.v1WallTimeS)}
          />
          <Stat
            label={`${data.summary.latestVersion.toUpperCase()} COST`}
            value={`$${data.summary.totalCostUsd.toFixed(2)}`}
            color="text-hud-amber"
          />
          <Stat
            label={`${data.summary.latestVersion.toUpperCase()} TIME`}
            value={formatTime(data.summary.totalWallTimeS)}
          />
          <div className="h-6 w-px bg-hud-border" />
          <Stat
            label="MODELS"
            value="gpt-5.3-codex / gpt-4.1-mini / qwen3-30b / moondream3"
            small
          />
          <div className="h-6 w-px bg-hud-border" />
          <Link
            href="/run"
            className="flex items-center gap-1.5 border border-foreground px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.2em] text-foreground transition-colors hover:bg-foreground hover:text-background"
          >
            <Play size={12} />
            Live Run
          </Link>
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  color,
  small,
}: {
  label: string;
  value: string;
  color?: string;
  small?: boolean;
}) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim">
        {label}
      </span>
      <span
        className={`${small ? "text-[12px]" : "text-xs"} font-bold tabular-nums ${color ?? "text-foreground"}`}
      >
        {value}
      </span>
    </div>
  );
}
