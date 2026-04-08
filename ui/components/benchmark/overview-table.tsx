"use client";

import { useState } from "react";
import type { BenchmarkPrompt, PromptResult } from "@/lib/types";
import { PromptDetail } from "./prompt-detail";

interface OverviewTableProps {
  prompts: BenchmarkPrompt[];
  latestVersion: string;
}

function scoreBadge(result: PromptResult | undefined) {
  if (!result || result.error) {
    return <span className="text-hud-red text-[10px]">ERROR</span>;
  }
  // Check if it was a forced answer (hit max iterations without critic pass)
  // We infer this from iterations === 20 and no explicit score stored
  const iters = result.iterations;
  if (iters >= 20) {
    return <span className="text-hud-amber text-[10px]">FORCED</span>;
  }
  // Otherwise show iteration count as a proxy for quality
  // Lower iterations = passed critic faster = better
  if (iters <= 5) {
    return <span className="text-hud-green text-[10px] font-bold">{iters} iters</span>;
  }
  if (iters <= 10) {
    return <span className="text-hud-green text-[10px]">{iters} iters</span>;
  }
  return <span className="text-hud-amber text-[10px]">{iters} iters</span>;
}

function deltaIndicator(v1: PromptResult | undefined, best: PromptResult | undefined, bestKey: string) {
  if (!v1 || !best || bestKey === "v1") {
    return <span className="text-hud-dim">&mdash;</span>;
  }
  if (best.error) {
    return <span className="text-hud-red text-[10px]">&darr;</span>;
  }
  // Compare iterations (lower is better)
  if (best.iterations < v1.iterations) {
    return <span className="text-hud-green text-[10px] font-bold">&uarr;</span>;
  }
  if (best.iterations > v1.iterations) {
    return <span className="text-hud-red text-[10px]">&darr;</span>;
  }
  return <span className="text-hud-dim">=</span>;
}

function formatName(name: string): string {
  return name.replace(/_/g, " ");
}

function formatTime(seconds: number): string {
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}

export function OverviewTable({ prompts, latestVersion }: OverviewTableProps) {
  const lv = latestVersion.toUpperCase();
  const [expandedId, setExpandedId] = useState<number | null>(null);

  return (
    <div className="border border-hud-border bg-hud-panel">
      {/* Header */}
      <div className="border-b border-hud-border px-3 py-2">
        <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-hud-label">
          Prompt Results
        </span>
      </div>

      {/* Table */}
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-hud-border text-[9px] font-bold uppercase tracking-[0.15em] text-hud-dim">
            <th className="px-3 py-2 text-right w-8">#</th>
            <th className="px-3 py-2 text-left">Prompt</th>
            <th className="px-3 py-2 text-left w-20">Video</th>
            <th className="px-3 py-2 text-center w-20">v1</th>
            <th className="px-3 py-2 text-right w-16">v1 Cost</th>
            <th className="px-3 py-2 text-right w-16">v1 Time</th>
            <th className="px-3 py-2 text-center w-20">{lv}</th>
            <th className="px-3 py-2 text-right w-16">{lv} Cost</th>
            <th className="px-3 py-2 text-right w-16">{lv} Time</th>
            <th className="px-3 py-2 text-center w-12">Ver</th>
            <th className="px-3 py-2 text-center w-10">&Delta;</th>
          </tr>
        </thead>
        <tbody>
          {prompts.map((prompt) => {
            const v1 = prompt.versions.v1;
            const best = prompt.versions[prompt.bestVersion];
            const isExpanded = expandedId === prompt.promptId;
            const hasMultipleVersions = Object.keys(prompt.versions).length > 1;

            return (
              <PromptRow
                key={prompt.promptId}
                prompt={prompt}
                v1={v1}
                best={best}
                isExpanded={isExpanded}
                hasMultipleVersions={hasMultipleVersions}
                onToggle={() =>
                  setExpandedId(isExpanded ? null : prompt.promptId)
                }
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PromptRow({
  prompt,
  v1,
  best,
  isExpanded,
  hasMultipleVersions,
  onToggle,
}: {
  prompt: BenchmarkPrompt;
  v1: PromptResult | undefined;
  best: PromptResult | undefined;
  isExpanded: boolean;
  hasMultipleVersions: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className={`border-b border-hud-border/50 cursor-pointer transition-colors ${
          isExpanded
            ? "bg-hud-border/20"
            : "hover:bg-hud-border/10"
        }`}
      >
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {prompt.promptId}
        </td>
        <td className="px-3 py-2 font-bold text-foreground">
          {formatName(prompt.promptName)}
        </td>
        <td className="px-3 py-2">
          <span className="text-[10px] text-hud-dim border border-hud-border px-1.5 py-0.5">
            {prompt.videoKey}
          </span>
        </td>
        <td className="px-3 py-2 text-center">{scoreBadge(v1)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {v1 && !v1.error ? `$${v1.costUsd.toFixed(2)}` : "\u2014"}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {v1 && !v1.error ? formatTime(v1.wallTimeS) : "\u2014"}
        </td>
        <td className="px-3 py-2 text-center">{scoreBadge(best)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-amber">
          {best && !best.error ? `$${best.costUsd.toFixed(2)}` : "\u2014"}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {best && !best.error ? formatTime(best.wallTimeS) : "\u2014"}
        </td>
        <td className="px-3 py-2 text-center">
          {hasMultipleVersions ? (
            <span className="text-[9px] font-bold text-hud-cyan border border-hud-cyan/30 px-1 py-0.5">
              {prompt.bestVersion.toUpperCase()}
            </span>
          ) : (
            <span className="text-hud-dim text-[10px]">v1</span>
          )}
        </td>
        <td className="px-3 py-2 text-center">
          {deltaIndicator(v1, best, prompt.bestVersion)}
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={11} className="p-0">
            <PromptDetail prompt={prompt} />
          </td>
        </tr>
      )}
    </>
  );
}
