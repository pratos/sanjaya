"use client";

import { useCallback, useRef, useState } from "react";
import type { BenchmarkPrompt } from "@/lib/types";
import { videoStreamUrl } from "@/lib/api";
import { AnswerRenderer, InlineTimestamps } from "./answer-renderer";
import { NarrativeNote } from "./narrative-note";
import { TraceTimeline } from "@/components/hud/trace-timeline";
import { NARRATIVES } from "@/lib/narratives";

const VIDEO_LABELS: Record<string, { title: string; channel: string; url: string }> = {
  podcast: { title: "RLM Theory Overview feat. Alex L. Zhang", channel: "Deep Learning with Yacine", url: "https://www.youtube.com/watch?v=5RAFKES5J6E" },
  curry: { title: "Stephen Curry's CRAZIEST Made Threes This Season!", channel: "NBA", url: "https://www.youtube.com/watch?v=zVg3FvMuDlw" },
  mkbhd: { title: "iPhone 17 Review: No Asterisks!", channel: "Marques Brownlee", url: "https://www.youtube.com/watch?v=rng_yUSwrgU" },
  football: { title: "Manchester City v Arsenal | FA Cup 2022-23", channel: "The Emirates FA Cup", url: "https://www.youtube.com/watch?v=9gyv2xh7qQw" },
  tech_talk: { title: "Prompting Is Becoming a Product Surface", channel: "Boundary", url: "https://www.youtube.com/watch?v=qdfwmYTO0Aw" },
  lvb_cooking: { title: "LongVideoBench — Cooking", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=1R5uPaL0V-0" },
  lvb_movie: { title: "LongVideoBench — Movie Recap", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=N7RTTiHsSjI" },
  lvb_travel: { title: "LongVideoBench — Travel", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=kOZnpwI2hIM" },
  lvb_history: { title: "LongVideoBench — History", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=fvCrE5NCsts" },
  lvb_art: { title: "LongVideoBench — Art", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=fZBC3nmvJb8" },
  lvb_geography: { title: "LongVideoBench — Geography", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=lzAESaVqix0" },
  lvb_stem: { title: "LongVideoBench — STEM", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=zda-T6wrEhs" },
  lvb_vlog: { title: "LongVideoBench — Life Vlog", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=Jfp1Ks7Hh1E" },
  lvb_napoleon: { title: "LongVideoBench — Napoleon", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=P9hDA0u6FO0" },
  lvb_dejavu: { title: "LongVideoBench — Deja Vu Stage", channel: "LongVideoBench", url: "https://www.youtube.com/watch?v=86CxyhFV9MI" },
};

interface PromptDetailProps {
  prompt: BenchmarkPrompt;
}

export function PromptDetail({ prompt }: PromptDetailProps) {
  const hiddenVersions = new Set(["v6", "v7"]);
  const allVersionKeys = Object.keys(prompt.versions).sort();
  const visibleVersionKeys = allVersionKeys.filter((v) => !hiddenVersions.has(v.toLowerCase()));
  const nonErrorVersionKeys = visibleVersionKeys.filter((v) => !prompt.versions[v]?.error);
  const versionKeys = nonErrorVersionKeys.length > 0 ? nonErrorVersionKeys : visibleVersionKeys;

  const [selectedVersion, setSelectedVersion] = useState(prompt.bestVersion);
  const activeVersion = versionKeys.includes(selectedVersion)
    ? selectedVersion
    : (versionKeys[0] ?? prompt.bestVersion);

  const videoRef = useRef<HTMLVideoElement>(null);
  const active = prompt.versions[activeVersion] ?? prompt.versions[prompt.bestVersion];
  const narrative = NARRATIVES[prompt.promptId];
  const traceEvents = active?.traceEvents ?? null;

  const handleSeek = useCallback((seconds: number) => {
    const video = videoRef.current;
    if (video) {
      video.currentTime = seconds;
      video.play().catch(() => {});
      video.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, []);

  if (!active) return null;

  return (
    <div className="bg-[#080808] border-t border-hud-border">
      {/* Version tabs (only if multiple versions) */}
      {versionKeys.length > 1 && (
        <div className="flex border-b border-hud-border">
          {versionKeys.map((v) => {
            const isActive = v === activeVersion;
            return (
              <button
                key={v}
                onClick={() => setSelectedVersion(v)}
                className={`px-4 py-1.5 text-[12px] font-bold uppercase tracking-[0.15em] transition-colors ${
                  isActive
                    ? "bg-hud-panel text-foreground border-b-2 border-hud-green"
                    : "text-hud-dim hover:text-foreground"
                }`}
              >
                {v.toUpperCase()}
              </button>
            );
          })}
        </div>
      )}

      <div className="p-4 space-y-4">
        {/* Narrative note */}
        {narrative && activeVersion !== "v1" && (
          <NarrativeNote
            title={narrative.title}
            text={narrative.text}
            improvement={narrative.improvement}
          />
        )}

        {/* Question */}
        <div>
          <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
            Question
          </span>
          <p className="text-[13px] text-hud-label leading-relaxed">
            {prompt.question}
          </p>
        </div>

        {/* Ground truth baseline */}
        {prompt.groundTruth && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
              Ground Truth
            </span>
            <p className="text-xs text-hud-green leading-relaxed">{prompt.groundTruth}</p>
          </div>
        )}

        {/* Answer summary */}
        {active.answerText && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
              Answer
            </span>
            <p className="text-xs text-foreground leading-relaxed whitespace-pre-wrap">
              <InlineTimestamps text={active.answerText} onSeek={handleSeek} />
            </p>
          </div>
        )}

        {/* Video + Metrics + Structured Data */}
        <div className="grid grid-cols-[200px_1fr] gap-4">
          {/* Video player spanning full width */}
          {prompt.videoPath && (
            <div className="col-span-2 mb-2">
              <span className="block text-[12px] text-hud-dim mb-1">
                {VIDEO_LABELS[prompt.videoKey] ? (
                  <a href={VIDEO_LABELS[prompt.videoKey].url} target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
                    <span className="text-foreground">{VIDEO_LABELS[prompt.videoKey].title}</span>
                    {" "}
                    <span className="text-hud-dim">by {VIDEO_LABELS[prompt.videoKey].channel}</span>
                  </a>
                ) : (
                  <span className="text-[13px] font-bold uppercase tracking-[0.15em]">Video &mdash; {prompt.videoKey}</span>
                )}
              </span>
              <video
                ref={videoRef}
                src={videoStreamUrl(prompt.videoPath)}
                controls
                preload="metadata"
                className="w-full max-h-[300px] border border-hud-border bg-black"
              />
            </div>
          )}
        </div>

        {/* Version comparison table */}
        {versionKeys.length > 1 && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">
              Version Comparison
            </span>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="border-b border-hud-border">
                    <th className="px-2 py-1 text-left text-[13px] font-bold uppercase tracking-wider text-hud-dim w-24">Metric</th>
                    {versionKeys.map((v) => (
                      <th key={v} className={`px-2 py-1 text-right text-[13px] font-bold uppercase tracking-wider ${v === prompt.bestVersion ? "text-hud-green" : "text-hud-dim"}`}>
                        {v.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <ComparisonRow label="Iterations" versions={versionKeys} data={prompt.versions} render={(d) => String(d.iterations)} better="lower" />
                  <ComparisonRow label="Cost" versions={versionKeys} data={prompt.versions} render={(d) => `$${d.costUsd.toFixed(4)}`} better="lower" valueColor="text-hud-amber" />
                  <ComparisonRow label="Wall Time" versions={versionKeys} data={prompt.versions} render={(d) => d.wallTimeS >= 60 ? `${(d.wallTimeS / 60).toFixed(1)}m` : `${d.wallTimeS.toFixed(1)}s`} better="lower" />
                  <ComparisonRow label="Tokens" versions={versionKeys} data={prompt.versions} render={(d) => `${(d.inputTokens / 1000).toFixed(1)}K / ${(d.outputTokens / 1000).toFixed(1)}K`} />
                  <ComparisonRow label="Evidence" versions={versionKeys} data={prompt.versions} render={(d) => String(d.evidenceCount)} better="higher" />
                  <ComparisonRow label="Subtitle" versions={versionKeys} data={prompt.versions} render={(d) => d.subtitle.subtitleSource} />
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Metrics (single version) + Structured Data side by side */}
        <div className="grid grid-cols-[200px_1fr] gap-4">
          {/* Metrics — only if single version */}
          {versionKeys.length <= 1 && (
            <div className="space-y-1.5 border-r border-hud-border pr-4">
              <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">
                Metrics
              </span>
              <MetricRow label="Iterations" value={String(active.iterations)} />
              <MetricRow
                label="Cost"
                value={`$${active.costUsd.toFixed(4)}`}
                color="text-hud-amber"
              />
              <MetricRow
                label="Tokens"
                value={`${(active.inputTokens / 1000).toFixed(1)}K / ${(active.outputTokens / 1000).toFixed(1)}K`}
              />
              <MetricRow
                label="Wall Time"
                value={`${active.wallTimeS.toFixed(1)}s`}
              />
              <MetricRow
                label="Evidence"
                value={String(active.evidenceCount)}
              />
              <MetricRow
                label="Subtitle"
                value={active.subtitle.subtitleSource}
              />
            </div>
          )}

          {/* Structured data */}
          <div className={versionKeys.length <= 1 ? "" : "col-span-2"}>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">
              Structured Data — {activeVersion.toUpperCase()}
            </span>
            <AnswerRenderer data={active.answerData} onSeek={handleSeek} />
          </div>
        </div>

        {/* Trace timeline (version-specific) */}
        {traceEvents && traceEvents.length > 0 && (
          <div className="h-80">
            <TraceTimeline
              events={traceEvents}
              startTime={traceEvents[0]?.timestamp ?? null}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function ComparisonRow({
  label,
  versions,
  data,
  render,
  better,
  valueColor,
}: {
  label: string;
  versions: string[];
  data: Record<string, import("@/lib/types").PromptResult>;
  render: (d: import("@/lib/types").PromptResult) => string;
  better?: "lower" | "higher";
  valueColor?: string;
}) {
  // Find the best value for highlighting
  let bestIdx: number | null = null;
  if (better) {
    let bestNum = better === "lower" ? Infinity : -Infinity;
    versions.forEach((v, i) => {
      const d2 = data[v];
      if (!d2 || d2.error) return;
      const raw = render(d2).replace(/[^0-9.]/g, "");
      const num = parseFloat(raw);
      if (isNaN(num)) return;
      if (better === "lower" ? num < bestNum : num > bestNum) {
        bestNum = num;
        bestIdx = i;
      }
    });
  }

  return (
    <tr className="border-b border-hud-border/50">
      <td className="px-2 py-1 text-[13px] uppercase tracking-wider text-hud-dim">{label}</td>
      {versions.map((v, i) => {
        const d2 = data[v];
        if (!d2 || d2.error) {
          return <td key={v} className="px-2 py-1 text-right text-hud-dim">—</td>;
        }
        const isBest = bestIdx === i;
        return (
          <td
            key={v}
            className={`px-2 py-1 text-right tabular-nums ${
              isBest ? "text-hud-green font-bold" : valueColor ?? "text-foreground"
            }`}
          >
            {render(d2)}
          </td>
        );
      })}
    </tr>
  );
}

function MetricRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-[13px] uppercase tracking-wider text-hud-dim">
        {label}
      </span>
      <span className={`text-[12px] tabular-nums ${color ?? "text-foreground"}`}>
        {value}
      </span>
    </div>
  );
}
