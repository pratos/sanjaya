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
  const latestVersion = data.summary.latestVersion.toUpperCase();

  return (
    <div className="border-b border-hud-border bg-hud-panel px-6 py-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-[0.28em] text-foreground lowercase">
            sanjaya
          </h1>
          <p className="mt-1 text-[12px] lowercase tracking-[0.12em] text-hud-dim">
            multi-modal analysis dashboard
          </p>
        </div>

        <Link
          href="/run"
          className="mt-1 flex items-center gap-1.5 whitespace-nowrap border border-foreground px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.12em] leading-none text-foreground transition-colors hover:bg-foreground hover:text-background"
        >
          <Play size={12} />
          Live Run
        </Link>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 lg:grid-cols-6">
        <StatCard label="Prompts" value={String(data.summary.totalPrompts)} />
        <StatCard label="Versions" value={data.summary.versions.join(", ")} />
        <StatCard label="V1 Cost" value={`$${data.summary.v1CostUsd.toFixed(2)}`} />
        <StatCard label="V1 Time" value={formatTime(data.summary.v1WallTimeS)} />
        <StatCard label={`${latestVersion} Cost`} value={`$${data.summary.totalCostUsd.toFixed(2)}`} valueClassName="text-hud-amber" />
        <StatCard label={`${latestVersion} Time`} value={formatTime(data.summary.totalWallTimeS)} />
      </div>

      <div className="mt-3 border-t border-hud-border/60 pt-3">
        <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-hud-dim">
          Models
        </span>
        <p className="mt-1 text-[13px] leading-relaxed text-foreground/90">
          gpt-5.3-codex / gpt-4.1-mini / qwen3-30b / moondream3
        </p>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  valueClassName,
}: {
  label: string;
  value: string;
  valueClassName?: string;
}) {
  return (
    <div className="border border-hud-border/70 bg-black/20 px-3 py-2">
      <span className="block text-[11px] font-bold uppercase tracking-[0.14em] text-hud-dim">
        {label}
      </span>
      <span className={`mt-1 block text-[20px] font-bold tabular-nums leading-none ${valueClassName ?? "text-foreground"}`}>
        {value}
      </span>
    </div>
  );
}
