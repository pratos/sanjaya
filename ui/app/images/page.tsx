"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Play } from "lucide-react";
import type { ImageBenchmarkData, ImagePrompt, ImageResult, TraceEvent } from "@/lib/types";
import { fetchImageBenchmarks } from "@/lib/api";
import { AnswerRenderer } from "@/components/benchmark/answer-renderer";
import { TraceTimeline } from "@/components/hud/trace-timeline";

function formatTime(seconds: number): string {
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}

function formatName(name: string): string {
  return name.replace(/_/g, " ");
}

function scoreBadge(result: ImageResult | undefined) {
  if (!result || result.error) {
    return <span className="text-hud-red text-[12px]">ERROR</span>;
  }
  const iters = result.iterations;
  if (iters >= 10) {
    return <span className="text-hud-amber text-[12px]">FORCED</span>;
  }
  if (iters <= 3) {
    return <span className="text-hud-green text-[12px] font-bold">{iters} iters</span>;
  }
  if (iters <= 6) {
    return <span className="text-hud-green text-[12px]">{iters} iters</span>;
  }
  return <span className="text-hud-amber text-[12px]">{iters} iters</span>;
}

function Stat({ label, value, color, small }: { label: string; value: string; color?: string; small?: boolean }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim">{label}</span>
      <span className={`${small ? "text-[12px]" : "text-xs"} font-bold tabular-nums ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-[13px] uppercase tracking-wider text-hud-dim">{label}</span>
      <span className={`text-[12px] tabular-nums ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

function ImageDetail({ prompt }: { prompt: ImagePrompt }) {
  const versionKeys = Object.keys(prompt.versions).sort();
  const [activeVersion, setActiveVersion] = useState(prompt.bestVersion);
  const active = prompt.versions[activeVersion];
  const traceEvents: TraceEvent[] | null = active?.traceEvents ?? null;

  if (!active) return null;

  return (
    <div className="bg-[#080808] border-t border-hud-border">
      {versionKeys.length > 1 && (
        <div className="flex border-b border-hud-border">
          {versionKeys.map((v) => (
            <button
              key={v}
              onClick={() => setActiveVersion(v)}
              className={`px-4 py-1.5 text-[12px] font-bold uppercase tracking-[0.15em] transition-colors ${
                v === activeVersion ? "bg-hud-panel text-foreground border-b-2 border-hud-green" : "text-hud-dim hover:text-foreground"
              }`}
            >
              {v.toUpperCase()}
              {v === prompt.bestVersion && <span className="ml-1.5 text-[12px] text-hud-green">BEST</span>}
            </button>
          ))}
        </div>
      )}

      <div className="p-4 space-y-4">
        <div>
          <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Question</span>
          <p className="text-[13px] text-hud-label leading-relaxed whitespace-pre-wrap">{prompt.question}</p>
        </div>

        {active.answerText && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Answer</span>
            <p className="text-xs text-foreground leading-relaxed whitespace-pre-wrap">{active.answerText}</p>
          </div>
        )}

        {active.groundTruth && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Ground Truth</span>
            <p className="text-xs text-hud-green leading-relaxed">{active.groundTruth}</p>
          </div>
        )}

        {active.evidenceSources.length > 0 && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Sources</span>
            <div className="flex flex-wrap gap-1.5">
              {active.evidenceSources.map((src) => (
                <span key={src} className="text-[12px] border border-hud-cyan/30 text-hud-cyan px-2 py-0.5">
                  {src.replace("image:", "")}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-[200px_1fr] gap-4">
          <div className="space-y-1.5 border-r border-hud-border pr-4">
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Metrics</span>
            <MetricRow label="Iterations" value={String(active.iterations)} />
            <MetricRow label="Cost" value={`$${active.costUsd.toFixed(4)}`} color="text-hud-amber" />
            <MetricRow label="Tokens" value={`${(active.inputTokens / 1000).toFixed(1)}K / ${(active.outputTokens / 1000).toFixed(1)}K`} />
            <MetricRow label="Wall Time" value={`${active.wallTimeS.toFixed(1)}s`} />
            <MetricRow label="Evidence" value={String(active.evidenceCount)} />
            <MetricRow label="Images" value={String(active.imagePaths.length)} />
          </div>

          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Structured Data</span>
            <AnswerRenderer data={active.answerData} onSeek={() => {}} />
          </div>
        </div>

        {traceEvents && traceEvents.length > 0 && (
          <div className="h-80">
            <TraceTimeline events={traceEvents} startTime={traceEvents[0]?.timestamp ?? null} />
          </div>
        )}
      </div>
    </div>
  );
}

export default function ImageBenchmarks() {
  const [data, setData] = useState<ImageBenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    fetchImageBenchmarks().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="border border-hud-red bg-hud-red/5 px-6 py-4">
          <span className="text-sm text-hud-red">{error}</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <span className="text-xs text-hud-dim uppercase tracking-[0.2em] animate-pulse">
          Loading image benchmarks...
        </span>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <div className="border-b border-hud-border bg-hud-panel px-6 py-3">
        <div className="flex items-center justify-end">
          <div className="flex items-center gap-8">
            <Stat label="PROMPTS" value={String(data.summary.totalPrompts)} />
            <Stat label="VERSIONS" value={data.summary.versions.join(", ")} />
            <Stat label="TOTAL COST" value={`$${data.summary.totalCostUsd.toFixed(2)}`} color="text-hud-amber" />
            <Stat label="TOTAL TIME" value={formatTime(data.summary.totalWallTimeS)} />
            <div className="h-6 w-px bg-hud-border" />
            <Stat label="MODELS" value="gpt-5.3-codex / gpt-4.1-mini" small />
            <div className="h-6 w-px bg-hud-border" />
            <Link
              href="/images/run"
              className="flex items-center gap-1.5 border border-foreground px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.2em] text-foreground transition-colors hover:bg-foreground hover:text-background"
            >
              <Play size={12} />
              Live Run
            </Link>
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 space-y-4">
        <div className="border border-hud-border bg-hud-panel">
          <div className="border-b border-hud-border px-3 py-2">
            <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-label">
              Image Analysis Results
            </span>
          </div>

          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-hud-border text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim">
                <th className="px-3 py-2 text-right w-8">#</th>
                <th className="px-3 py-2 text-left">Prompt</th>
                <th className="px-3 py-2 text-center w-20">Score</th>
                <th className="px-3 py-2 text-right w-16">Cost</th>
                <th className="px-3 py-2 text-right w-16">Time</th>
                <th className="px-3 py-2 text-right w-16">Tokens</th>
                <th className="px-3 py-2 text-center w-16">Evidence</th>
                <th className="px-3 py-2 text-center w-16">Images</th>
              </tr>
            </thead>
            <tbody>
              {data.prompts.map((prompt) => {
                const best = prompt.versions[prompt.bestVersion];
                const isExpanded = expandedId === prompt.promptId;
                return (
                  <PromptRow
                    key={prompt.promptId}
                    prompt={prompt}
                    best={best}
                    isExpanded={isExpanded}
                    onToggle={() => setExpandedId(isExpanded ? null : prompt.promptId)}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PromptRow({
  prompt,
  best,
  isExpanded,
  onToggle,
}: {
  prompt: ImagePrompt;
  best: ImageResult | undefined;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className={`border-b border-hud-border/50 cursor-pointer transition-colors ${
          isExpanded ? "bg-hud-border/20" : "hover:bg-hud-border/10"
        }`}
      >
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">{prompt.promptId}</td>
        <td className="px-3 py-2 font-bold text-foreground">{formatName(prompt.promptName)}</td>
        <td className="px-3 py-2 text-center">{scoreBadge(best)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-amber">
          {best && !best.error ? `$${best.costUsd.toFixed(2)}` : "\u2014"}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {best && !best.error ? formatTime(best.wallTimeS) : "\u2014"}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {best && !best.error ? `${(best.inputTokens / 1000).toFixed(0)}K` : "\u2014"}
        </td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-dim">
          {best && !best.error ? best.evidenceCount : "\u2014"}
        </td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-dim">
          {best && !best.error ? best.imagePaths.length : "\u2014"}
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={8} className="p-0">
            <ImageDetail prompt={prompt} />
          </td>
        </tr>
      )}
    </>
  );
}
