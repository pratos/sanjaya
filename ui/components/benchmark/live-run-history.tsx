"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import type { LiveRunsData, LiveRunItem, PromptResult, VideoInfo } from "@/lib/types";
import { PromptDetail } from "./prompt-detail";

interface LiveRunHistoryProps {
  data: LiveRunsData;
  videos: VideoInfo[];
}

function scoreBadge(result: PromptResult | undefined) {
  if (!result || result.error) {
    return <span className="text-hud-red text-[10px]">ERROR</span>;
  }
  const iters = result.iterations;
  if (iters >= 20) {
    return <span className="text-hud-amber text-[10px]">FORCED</span>;
  }
  if (iters <= 5) {
    return <span className="text-hud-green text-[10px] font-bold">{iters} iters</span>;
  }
  if (iters <= 10) {
    return <span className="text-hud-green text-[10px]">{iters} iters</span>;
  }
  return <span className="text-hud-amber text-[10px]">{iters} iters</span>;
}

function formatTime(seconds: number): string {
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}

function formatTimestamp(ts: string): string {
  // "20260408-165850" → "Apr 8, 16:58"
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  if (!m) return ts;
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const month = months[parseInt(m[2], 10) - 1] ?? m[2];
  const day = parseInt(m[3], 10);
  return `${month} ${day}, ${m[4]}:${m[5]}`;
}

function shortModel(model: string): string {
  const parts = model.split("/");
  return parts[parts.length - 1];
}

export function LiveRunHistory({ data, videos }: LiveRunHistoryProps) {
  const [collapsed, setCollapsed] = useState(true);
  const [selectedRun, setSelectedRun] = useState<LiveRunItem | null>(null);

  if (data.totalRuns === 0) return null;

  // Show newest first
  const runs = [...data.runs].reverse();

  return (
    <>
      <div className="border border-hud-border bg-hud-panel">
        {/* Collapsible header */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-between border-b border-hud-border px-3 py-2 hover:bg-hud-border/10 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-hud-label">
              Live Run History
            </span>
            <span className="text-[10px] text-hud-dim tabular-nums">
              {data.totalRuns} runs
            </span>
            <span className="text-[10px] text-hud-dim tabular-nums">
              ${data.totalCostUsd.toFixed(2)} total
            </span>
            <span className="text-[10px] text-hud-dim tabular-nums">
              {formatTime(data.totalWallTimeS)} total
            </span>
          </div>
          <span className="text-[10px] text-hud-dim">
            {collapsed ? "+" : "\u2212"}
          </span>
        </button>

        {/* Videos bar */}
        {!collapsed && videos.length > 0 && (
          <div className="border-b border-hud-border px-3 py-1.5 flex items-center gap-2">
            <span className="text-[9px] uppercase tracking-[0.15em] text-hud-dim">Videos:</span>
            {videos.map((v) => (
              <span
                key={v.key}
                className="text-[10px] text-hud-dim border border-hud-border px-1.5 py-0.5"
              >
                {v.key}
              </span>
            ))}
          </div>
        )}

        {/* Table */}
        {!collapsed && (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-hud-border text-[9px] font-bold uppercase tracking-[0.15em] text-hud-dim">
                <th className="px-3 py-2 text-right w-8">#</th>
                <th className="px-3 py-2 text-left w-24">When</th>
                <th className="px-3 py-2 text-left">Question</th>
                <th className="px-3 py-2 text-left w-28">Model</th>
                <th className="px-3 py-2 text-center w-20">Score</th>
                <th className="px-3 py-2 text-right w-16">Cost</th>
                <th className="px-3 py-2 text-right w-16">Time</th>
                <th className="px-3 py-2 text-right w-20">Tokens</th>
                <th className="px-3 py-2 text-right w-12">Evid.</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run, idx) => {
                const result = run.prompt.versions.live ?? run.prompt.versions[run.prompt.bestVersion];
                return (
                  <tr
                    key={run.runId}
                    onClick={() => setSelectedRun(run)}
                    className="border-b border-hud-border/50 cursor-pointer transition-colors hover:bg-hud-border/10"
                  >
                    <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
                      {runs.length - idx}
                    </td>
                    <td className="px-3 py-2 text-[10px] text-hud-dim tabular-nums">
                      {formatTimestamp(run.timestamp)}
                    </td>
                    <td className="px-3 py-2 font-bold text-foreground truncate max-w-[400px]">
                      {run.prompt.question}
                    </td>
                    <td className="px-3 py-2 text-[10px] text-hud-dim">
                      {shortModel(run.model)}
                    </td>
                    <td className="px-3 py-2 text-center">{scoreBadge(result)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-hud-amber">
                      {result && !result.error ? `$${result.costUsd.toFixed(2)}` : "\u2014"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
                      {result && !result.error ? formatTime(result.wallTimeS) : "\u2014"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
                      {result && !result.error
                        ? `${((result.inputTokens + result.outputTokens) / 1000).toFixed(1)}k`
                        : "\u2014"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
                      {result && !result.error ? result.evidenceCount : "\u2014"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Detail modal */}
      {selectedRun && (
        <LiveRunModal run={selectedRun} onClose={() => setSelectedRun(null)} />
      )}
    </>
  );
}

function LiveRunModal({ run, onClose }: { run: LiveRunItem; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const result = run.prompt.versions.live ?? run.prompt.versions[run.prompt.bestVersion];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative flex h-[90vh] w-[85vw] max-w-5xl flex-col border border-hud-border bg-[#0a0a0a]">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-hud-border px-5 py-3 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-hud-green">
              LIVE RUN
            </span>
            <span className="text-hud-dim">·</span>
            <span className="font-mono text-xs text-hud-dim">{run.runId}</span>
            <span className="text-hud-dim">·</span>
            <span className="text-xs text-hud-dim">{formatTimestamp(run.timestamp)}</span>
            <span className="text-hud-dim">·</span>
            <span className="text-xs text-hud-dim">{shortModel(run.model)}</span>
            {result && !result.error && (
              <>
                <span className="text-hud-dim">·</span>
                <span className="text-xs text-hud-amber">${result.costUsd.toFixed(4)}</span>
                <span className="text-hud-dim">·</span>
                <span className="text-xs text-hud-dim">{formatTime(result.wallTimeS)}</span>
              </>
            )}
          </div>
          <button onClick={onClose} className="text-hud-dim hover:text-foreground">
            <X size={16} />
          </button>
        </div>

        {/* Body — reuse PromptDetail */}
        <div className="flex-1 overflow-y-auto">
          <PromptDetail prompt={run.prompt} />
        </div>
      </div>
    </div>
  );
}
